use crate::graphviz;
use rand::Rng;
use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

/// Operation represents the operations that can be performed in the computational graph.
#[allow(dead_code)]
pub enum Operation {
    None,
    Add(AutogradValue, AutogradValue),
    Multiply(AutogradValue, AutogradValue),
    Power(AutogradValue, f64),
    Relu(AutogradValue),
    Negate(AutogradValue),
    Exp(AutogradValue),
}

/// ValueData contains the actual data, gradient, and operation information for an AutogradValue.
pub struct ValueData {
    data: f64,
    grad: f64,
    op: Operation,
}

impl ValueData {
    pub fn op(&self) -> &Operation {
        &self.op
    }
}

impl fmt::Debug for ValueData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ValueData")
            .field("data", &self.data)
            .field("grad", &self.grad)
            .finish()
    }
}

/// AutogradValue represents a value in the computational graph that tracks gradients.
#[derive(Clone)]
pub struct AutogradValue(pub(crate) Rc<RefCell<ValueData>>);

impl fmt::Debug for AutogradValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AutogradValue")
            .field("data", &self.data())
            .field("grad", &self.grad())
            .finish()
    }
}

impl AutogradValue {
    pub fn new(data: f64) -> Self {
        Self(Rc::new(RefCell::new(ValueData {
            data,
            grad: 0.0,
            op: Operation::None,
        })))
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    fn set_data(&self, value: f64) {
        self.0.borrow_mut().data = value;
    }

    fn set_grad(&self, value: f64) {
        self.0.borrow_mut().grad = value;
    }

    fn add_grad(&self, value: f64) {
        self.0.borrow_mut().grad += value;
    }

    pub fn zero_grad(&self) {
        self.set_grad(0.0);
    }

    // Build a topological sort of the computational graph
    // Computing gradients in reverse topological order, ensuring each node's gradient is computed
    // only once after all its dependent nodes are processed
    fn build_topo(&self) -> Vec<AutogradValue> {
        // NOTE: For efficiency in a production system, this topological sort
        // could be cached if the graph structure doesn't change between backward passes.
        let mut topo = Vec::new();
        let mut visited = HashSet::new();

        fn build_topo_recursive(
            v: &AutogradValue,
            topo: &mut Vec<AutogradValue>,
            visited: &mut HashSet<*const ()>,
        ) {
            // NOTE: Using pointer address as unique identifier works for this educational implementation
            // but could be problematic in production code if memory is reallocated.
            let ptr = Rc::as_ptr(&v.0) as *const ();

            if visited.insert(ptr) {
                match &v.0.borrow().op {
                    Operation::None => {}
                    Operation::Add(a, b) => {
                        build_topo_recursive(a, topo, visited);
                        build_topo_recursive(b, topo, visited);
                    }
                    Operation::Multiply(a, b) => {
                        build_topo_recursive(a, topo, visited);
                        build_topo_recursive(b, topo, visited);
                    }
                    Operation::Power(a, _) => {
                        build_topo_recursive(a, topo, visited);
                    }
                    Operation::Relu(a) => {
                        build_topo_recursive(a, topo, visited);
                    }
                    Operation::Negate(a) => {
                        build_topo_recursive(a, topo, visited);
                    }
                    Operation::Exp(a) => {
                        build_topo_recursive(a, topo, visited);
                    }
                }
                // Add node to topo order after all its dependencies
                topo.push(v.clone());
            }
        }

        build_topo_recursive(self, &mut topo, &mut visited);
        topo
    }

    /// Backward method using topological sorting
    pub fn backward(&self, gradient: f64) {
        self.add_grad(gradient);

        let topo = self.build_topo();

        // Process nodes in reverse topological order
        for node in topo.iter().rev() {
            let grad = node.grad();

            match &node.0.borrow().op {
                Operation::None => {
                    // Leaf node, nothing to do
                }
                Operation::Add(a, b) => {
                    // Chain rule for addition
                    a.add_grad(grad);
                    b.add_grad(grad);
                }
                Operation::Multiply(a, b) => {
                    // Chain rule for multiplication
                    let a_val = a.data();
                    let b_val = b.data();
                    a.add_grad(grad * b_val);
                    b.add_grad(grad * a_val);
                }
                Operation::Power(a, n) => {
                    // Chain rule for power
                    let a_val = a.data();
                    a.add_grad(grad * n * a_val.powf(n - 1.0));
                }
                Operation::Relu(a) => {
                    // Chain rule for ReLU
                    let a_val = a.data();
                    if a_val > 0.0 {
                        a.add_grad(grad);
                    }
                }
                Operation::Negate(a) => {
                    // Chain rule for negation
                    a.add_grad(-grad);
                }
                Operation::Exp(a) => {
                    // Chain rule for exponential
                    a.add_grad(grad * node.data());
                }
            }
        }
    }

    pub fn pow(&self, exponent: f64) -> AutogradValue {
        let a = self.clone();

        AutogradValue(Rc::new(RefCell::new(ValueData {
            data: self.data().powf(exponent),
            grad: 0.0,
            op: Operation::Power(a, exponent),
        })))
    }

    pub fn relu(&self) -> AutogradValue {
        let a = self.clone();
        let data = self.data();

        AutogradValue(Rc::new(RefCell::new(ValueData {
            data: if data > 0.0 { data } else { 0.0 },
            grad: 0.0,
            op: Operation::Relu(a),
        })))
    }

    #[allow(dead_code)]
    pub fn exp(&self) -> AutogradValue {
        let a = self.clone();
        let data = self.data();

        AutogradValue(Rc::new(RefCell::new(ValueData {
            data: data.exp(),
            grad: 0.0,
            op: Operation::Exp(a),
        })))
    }
}

impl<'b> Add<&'b AutogradValue> for &AutogradValue {
    type Output = AutogradValue;

    fn add(self, rhs: &'b AutogradValue) -> Self::Output {
        let a = self.clone();
        let b = rhs.clone();

        AutogradValue(Rc::new(RefCell::new(ValueData {
            data: self.data() + rhs.data(),
            grad: 0.0,
            op: Operation::Add(a, b),
        })))
    }
}

impl Add<f64> for &AutogradValue {
    type Output = AutogradValue;

    fn add(self, rhs: f64) -> Self::Output {
        let rhs_value = AutogradValue::new(rhs);
        self + &rhs_value
    }
}

impl Add<&AutogradValue> for f64 {
    type Output = AutogradValue;

    fn add(self, rhs: &AutogradValue) -> Self::Output {
        let lhs_value = AutogradValue::new(self);
        &lhs_value + rhs
    }
}

impl<'b> Mul<&'b AutogradValue> for &AutogradValue {
    type Output = AutogradValue;

    fn mul(self, rhs: &'b AutogradValue) -> Self::Output {
        let a = self.clone();
        let b = rhs.clone();

        AutogradValue(Rc::new(RefCell::new(ValueData {
            data: self.data() * rhs.data(),
            grad: 0.0,
            op: Operation::Multiply(a, b),
        })))
    }
}

impl Mul<f64> for &AutogradValue {
    type Output = AutogradValue;

    fn mul(self, rhs: f64) -> Self::Output {
        let rhs_value = AutogradValue::new(rhs);
        self * &rhs_value
    }
}

impl Mul<&AutogradValue> for f64 {
    type Output = AutogradValue;

    fn mul(self, rhs: &AutogradValue) -> Self::Output {
        let lhs_value = AutogradValue::new(self);
        &lhs_value * rhs
    }
}

impl Neg for &AutogradValue {
    type Output = AutogradValue;

    fn neg(self) -> Self::Output {
        let a = self.clone();

        AutogradValue(Rc::new(RefCell::new(ValueData {
            data: -self.data(),
            grad: 0.0,
            op: Operation::Negate(a),
        })))
    }
}

impl<'b> Sub<&'b AutogradValue> for &AutogradValue {
    type Output = AutogradValue;

    fn sub(self, rhs: &'b AutogradValue) -> Self::Output {
        self + &(-rhs)
    }
}

impl Sub<f64> for &AutogradValue {
    type Output = AutogradValue;

    fn sub(self, rhs: f64) -> Self::Output {
        let rhs_value = AutogradValue::new(rhs);
        self - &rhs_value
    }
}

impl Sub<&AutogradValue> for f64 {
    type Output = AutogradValue;

    fn sub(self, rhs: &AutogradValue) -> Self::Output {
        let lhs_value = AutogradValue::new(self);
        &lhs_value - rhs
    }
}

impl<'b> Div<&'b AutogradValue> for &AutogradValue {
    type Output = AutogradValue;

    fn div(self, rhs: &'b AutogradValue) -> Self::Output {
        self * &rhs.pow(-1.0)
    }
}

impl Div<f64> for &AutogradValue {
    type Output = AutogradValue;

    fn div(self, rhs: f64) -> Self::Output {
        let rhs_value = AutogradValue::new(rhs);
        self / &rhs_value
    }
}

impl Div<&AutogradValue> for f64 {
    type Output = AutogradValue;

    fn div(self, rhs: &AutogradValue) -> Self::Output {
        let lhs_value = AutogradValue::new(self);
        &lhs_value / rhs
    }
}

impl From<f64> for AutogradValue {
    fn from(value: f64) -> Self {
        AutogradValue::new(value)
    }
}

#[derive(Copy, Clone)]
pub enum Activation {
    ReLU,
    Linear,
}

pub trait Parameters {
    fn parameters(&self) -> Vec<AutogradValue>;
    fn parameter_count(&self) -> usize {
        self.parameters().len()
    }
}

pub trait Forward {
    /// Process inputs and return all outputs of the model
    fn forward(&self, x: &[AutogradValue]) -> Vec<AutogradValue>;

    /// Convenience method to get only the first output (if any)
    fn forward_single(&self, x: &[AutogradValue]) -> Option<AutogradValue> {
        self.forward(x).first().cloned()
    }
}

pub trait Module: Parameters + Forward {}
impl<T> Module for T where T: Parameters + Forward {}

struct Neuron {
    w: Vec<AutogradValue>,
    b: AutogradValue,
    activation: Activation,
}

impl Neuron {
    fn new(nin: usize, activation: Activation) -> Self {
        let mut rng = rand::rng();
        // Kaiming initialization helps mitigate the vanishing/exploding gradient problem in deep networks,
        // particularly those using ReLU activations. Traditional methods like random normal or Xavier
        // initialization may cause gradients to shrink or grow uncontrollably during backpropagation.
        // Kaiming initialization samples weights from a Gaussian distribution with mean 0 and
        // standard deviation sqrt(2/n), where n is the number of input connections to the node.
        let w = (0..nin)
            .map(|_| AutogradValue::new(rng.random_range(-1.0..1.0) * (2.0 / nin as f64).sqrt()))
            .collect();
        // To prevent the dying ReLU problem, biases for neurons with ReLU activation
        // can be initialized to a small positive value (e.g., 0.1). This increases
        // the likelihood of nonzero gradients at initialization.
        let b = AutogradValue::new(0.1);

        Self { w, b, activation }
    }
}

impl Parameters for Neuron {
    fn parameters(&self) -> Vec<AutogradValue> {
        let mut params = self.w.clone();
        params.push(self.b.clone());
        params
    }
}

impl Forward for Neuron {
    fn forward(&self, x: &[AutogradValue]) -> Vec<AutogradValue> {
        assert_eq!(x.len(), self.w.len());

        // w dot x + b
        // NOTE: Would be more efficient to implement an efficient dot product producing AutogradValue
        let act = self
            .w
            .iter()
            .zip(x.iter())
            .fold(self.b.clone(), |acc, (wi, xi)| &(wi * xi) + &acc);

        match self.activation {
            Activation::ReLU => {
                vec![act.relu()]
            }
            Activation::Linear => {
                vec![act]
            }
        }
    }
}

impl fmt::Display for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.activation {
            Activation::ReLU => {
                write!(f, "ReLUNeuron({})", self.w.len())
            }
            Activation::Linear => {
                write!(f, "LinearNeuron({})", self.w.len())
            }
        }
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize, activation: Activation) -> Self {
        let neurons = (0..nout).map(|_| Neuron::new(nin, activation)).collect();
        Self { neurons }
    }
}

impl Parameters for Layer {
    fn parameters(&self) -> Vec<AutogradValue> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

impl Forward for Layer {
    fn forward(&self, x: &[AutogradValue]) -> Vec<AutogradValue> {
        self.neurons.iter().flat_map(|n| n.forward(x)).collect()
    }
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Layer of [")?;
        for (i, n) in self.neurons.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", n)?;
        }
        write!(f, "]")
    }
}

pub struct Mlp {
    layers: Vec<Layer>,
}

impl Parameters for Mlp {
    fn parameters(&self) -> Vec<AutogradValue> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }
}

impl Forward for Mlp {
    fn forward(&self, x: &[AutogradValue]) -> Vec<AutogradValue> {
        let mut current_input = x.to_vec();

        for layer in &self.layers {
            current_input = layer.forward(&current_input);
        }

        current_input
    }
}

pub struct MlpBuilder {
    input_size: usize,
    layers: Vec<(usize, Activation)>,
}

impl MlpBuilder {
    pub fn new(input_size: usize) -> Self {
        Self {
            input_size,
            layers: Vec::new(),
        }
    }

    pub fn add_layer(mut self, size: usize, activation: Activation) -> Self {
        self.layers.push((size, activation));
        self
    }

    pub fn build(self) -> Mlp {
        let mut layers = Vec::new();
        let mut prev_size = self.input_size;

        for (size, activation) in self.layers.into_iter() {
            layers.push(Layer::new(prev_size, size, activation));
            prev_size = size;
        }

        Mlp { layers }
    }
}

impl fmt::Display for Mlp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MLP of [")?;
        for (i, layer) in self.layers.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", layer)?;
        }
        write!(f, "]")
    }
}

pub trait Loss {
    fn calculate(&self, predictions: &[AutogradValue], targets: &[AutogradValue]) -> AutogradValue;
}

pub struct MseLoss {}
impl Loss for MseLoss {
    fn calculate(&self, predictions: &[AutogradValue], targets: &[AutogradValue]) -> AutogradValue {
        assert_eq!(
            predictions.len(),
            targets.len(),
            "Predictions and targets must have the same length"
        );

        if predictions.is_empty() {
            return AutogradValue::new(0.0);
        }

        let mut sum_squared_error = AutogradValue::new(0.0);
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let diff = pred - target;
            let squared = &diff.pow(2.0);
            sum_squared_error = &sum_squared_error + squared;
        }

        // Calculate mean by dividing by the number of examples
        &sum_squared_error / predictions.len() as f64
    }
}

pub trait Optimizer {
    fn step(&self, params: &[AutogradValue]);
    fn zero_grad(&self, params: &[AutogradValue]) {
        for p in params {
            p.zero_grad();
        }
    }
}

pub struct GradientDescent {
    learning_rate: f64,
}

impl GradientDescent {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for GradientDescent {
    fn step(&self, params: &[AutogradValue]) {
        for p in params {
            p.set_data(p.data() - self.learning_rate * p.grad());
        }
    }
}

pub struct Trainer<M: Module, O: Optimizer, L: Loss> {
    model: M,
    optimizer: O,
    loss: L,
}

impl<M: Module, O: Optimizer, L: Loss> Trainer<M, O, L> {
    pub fn new(model: M, optimizer: O, loss: L) -> Self {
        Self {
            model,
            optimizer,
            loss,
        }
    }

    pub fn train(
        &self,
        x: &[&[AutogradValue]],
        y: &[AutogradValue],
        epochs: usize,
    ) -> Result<Vec<f64>, ModelError> {
        if x.len() != y.len() {
            return Err(ModelError::InputSizeMismatch {
                input: x.len(),
                target: y.len(),
            });
        }

        let mut loss_history = Vec::new();

        if epochs == 0 {
            return Ok(loss_history);
        }

        for epoch in 0..epochs {
            let mut predictions = Vec::with_capacity(x.len());
            for (idx, &input) in x.iter().enumerate() {
                let output = self.model.forward_single(input);
                if let Some(val) = output {
                    predictions.push(val);
                } else {
                    eprintln!(
                        "Error: Model produced empty output for input at index {}",
                        idx
                    );
                    return Err(ModelError::EmptyPredictions);
                }
            }

            if predictions.is_empty() {
                eprintln!(
                    "Error: No valid predictions were generated in epoch {}",
                    epoch
                );
                return Err(ModelError::EmptyPredictions);
            }

            let loss = self.loss.calculate(&predictions, y);
            if epoch == epochs - 1 {
                graphviz::dot_to_file_with_model(&loss, &self.model, x);
            }

            // By default, gradients are accumulated on subsequent backward passes
            // This requires to explicitly reset the gradients, but can limit the scope to optimized
            // parameters only. The computational graph is recreated during each forward pass, so
            // intermediate `AutogradValue` objects are recreated (with 0.0 default gradient), hence
            // their previous gradients are automatically discarded
            self.optimizer.zero_grad(&self.model.parameters());
            // Optional: Zero input/target gradients
            for batch in x {
                for input in *batch {
                    input.zero_grad();
                }
            }
            for target in y {
                target.zero_grad();
            }
            // Backward pass
            // Before calling loss.backward(1.0), all gradients must be 0.0.
            // Parameter gradients are 0.0 because they are explicitly zeroed with zero_grad()
            // Intermediate values have 0.0 gradients as they're newly created during the forward pass
            // Input/targets are not recreated and are not parameters. They can be ignored or
            // optionally also be set to zero
            loss.backward(1.0);

            // Update parameters
            self.optimizer.step(&self.model.parameters());

            let report_interval = if epochs >= 100 { epochs / 100 } else { 1 };
            if epoch % report_interval == 0 || epoch == epochs - 1 {
                loss_history.push(loss.data());
                println!("Epoch {}: Loss = {}", epoch, loss.data());
            }
        }

        Ok(loss_history)
    }

    pub fn predict(&self, x: &[AutogradValue]) -> Vec<f64> {
        self.model.forward(x).iter().map(|val| val.data()).collect()
    }

    pub fn predict_single(&self, x: &[AutogradValue]) -> Option<f64> {
        self.model.forward_single(x).map(|val| val.data())
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ModelError {
    #[error("input size mismatch (input {input:?}, target {target:?})")]
    InputSizeMismatch { input: usize, target: usize },
    #[error("empty predictions")]
    EmptyPredictions,
}
