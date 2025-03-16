# RustGrad: Neural Networks with Automatic Differentiation in Rust

A from-scratch implementation of neural networks with automatic differentiation in pure Rust.

## 🚧 Educational Project Disclaimer

This is a **personal educational implementation** for understanding of deep learning frameworks. It is **not production-ready** and lacks the optimizations and extensive testing of frameworks like PyTorch or TensorFlow.

## 💡 Key Technical Accomplishments

- **Automatic Differentiation Engine** that builds computational graphs and computes exact gradients
- **Neural Network Framework** with neurons, layers, and multi-layer perceptron architectures
- **Optimization System** for training neural networks with gradient descent

## ✨ Features

- **Automatic Differentiation**: Tracks gradients through operations like addition, multiplication, power, ReLU, etc.
- **Neural Network Components**: Neurons, layers, and MLPs with configurable architectures
- **Training Infrastructure**: Loss functions (MSE), optimizers (SGD), and training loop
- **Visualization**: Graph visualization using GraphViz

## 🧠 Implementation Highlights

### Computational Graph and Automatic Differentiation

```rust
// Operations automatically build a computational graph
let a = AutogradValue::new(2.0);
let b = AutogradValue::new(3.0);
let c = &a + &b;           // Addition operation
let d = &c * &a;           // Multiplication operation
let e = d.pow(2.0);        // Power operation
e.backward(1.0);           // Backward pass computes all gradients
```

### Neural Network API

```rust
// Create a neural network with one hidden layer
let model = MlpBuilder::new(2)  // 2 input features
    .add_layer(8, Activation::ReLU)  // Hidden layer with 8 neurons
    .add_layer(4, Activation::ReLU)  // Hidden layer with 4 neurons
    .add_layer(1, Activation::Linear)  // Output layer with 1 neuron
    .build();

// Setup training components
let optimizer = GradientDescent::new(0.1);  // Learning rate = 0.1
let loss_fn = MseLoss{};
let trainer = Trainer::new(model, optimizer, loss_fn);

// Train and predict
trainer.train(&inputs, &targets, 1000).expect("Training failed");
let prediction = trainer.predict_single(&input).unwrap();
```

## 🔧 Technical Implementation Details

- **Reverse-mode Automatic Differentiation** with topological sorting of the computational graph
- **Kaiming/He Initialization** for neural network weights to improve training stability
- **Smart Pointer Usage** (`Rc<RefCell<>>`) to handle shared mutable state in the computational graph
- **Trait-based Design** for components like modules, optimizers, and loss functions
- **Safe Error Handling** with custom error types using thiserror

## 🎓 Learning Outcomes

Building this project provided deep insights into:

1. The inner workings of deep learning frameworks
2. Implementation challenges of automatic differentiation
3. Managing complex data structures with Rust's ownership system
4. Designing ergonomic APIs for mathematical operations

## 📊 Example: 3D Parity Function (3D XOR)

The classic XOR problem demonstrates a neural network's ability to learn non-linear patterns.
The 3D parity function extends this concept to three dimensions. It outputs 1 if the number of positive inputs is odd and -1 if it is even.

```rust
// Define Model
let model = MlpBuilder::new(3)  // 3 input features
    .add_layer(8, Activation::ReLU)  // Hidden layer with 8 neurons
    .add_layer(8, Activation::ReLU)  // Second hidden layer with 8 neurons
    .add_layer(1, Activation::Linear)  // Output layer with 1 neuron
    .build();

// Create a dataset for 3D XOR (parity function)
let x_data = [
    [(-1.0).into(), (-1.0).into(), (-1.0).into()],  // 0 positive inputs -> even
    [(-1.0).into(), (-1.0).into(), 1.0.into()],     // 1 positive input -> odd
    [(-1.0).into(), 1.0.into(), (-1.0).into()],     // 1 positive input -> odd
    [(-1.0).into(), 1.0.into(), 1.0.into()],        // 2 positive inputs -> even
    [1.0.into(), (-1.0).into(), (-1.0).into()],     // 1 positive input -> odd
    [1.0.into(), (-1.0).into(), 1.0.into()],        // 2 positive inputs -> even
    [1.0.into(), 1.0.into(), (-1.0).into()],        // 2 positive inputs -> even
    [1.0.into(), 1.0.into(), 1.0.into()],           // 3 positive inputs -> odd
];

let y_data = [
    (-1.0).into(),  // even -> -1
    1.0.into(),     // odd -> 1
    1.0.into(),     // odd -> 1
    (-1.0).into(),  // even -> -1
    1.0.into(),     // odd -> 1
    (-1.0).into(),  // even -> -1
    (-1.0).into(),  // even -> -1
    1.0.into(),     // odd -> 1
];

// Train the model
let trainer = Trainer::new(model, GradientDescent::new(0.01), MseLoss{});
let x_refs: Vec<&[AutogradValue]> = x_data.iter().map(|x| x.as_slice()).collect();
trainer.train(&x_refs, &y_data, 10_000).expect("Training failed");

// Make predictions
for x in x_refs {
    println!("Prediction: {:?}", trainer.predict_single(x));
}
```

## 🔮 Potential Enhancements

- Improved ownership model with node ids over addresses
- Additional optimizers (Adam, RMSProp)
- More layer types and activation functions
- SIMD and GPU acceleration
- Batch processing and performance optimizations
- Comprehensive testing suite
- Efficiency (clone usage, dot product, ...)
- Much more...

## 🌟 Inspiration

This project was inspired by:
- Andrej Karpathy's "[micrograd](https://github.com/karpathy/micrograd)"