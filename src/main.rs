use rust_grad::autograd::{
    Activation, AutogradValue, GradientDescent, MlpBuilder, MseLoss, Parameters, Trainer,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define Model
    let model = MlpBuilder::new(3) // 3 input features
        .add_layer(8, Activation::ReLU) // Hidden layer with 8 neurons
        .add_layer(8, Activation::ReLU) // Second hidden layer with 8 neurons
        .add_layer(1, Activation::Linear) // Output layer with 1 neuron
        .build();

    // Create a dataset for 3D XOR (parity function)
    let x_data = [
        [(-1.0).into(), (-1.0).into(), (-1.0).into()], // 0 positive inputs -> even
        [(-1.0).into(), (-1.0).into(), 1.0.into()],    // 1 positive input -> odd
        [(-1.0).into(), 1.0.into(), (-1.0).into()],    // 1 positive input -> odd
        [(-1.0).into(), 1.0.into(), 1.0.into()],       // 2 positive inputs -> even
        [1.0.into(), (-1.0).into(), (-1.0).into()],    // 1 positive input -> odd
        [1.0.into(), (-1.0).into(), 1.0.into()],       // 2 positive inputs -> even
        [1.0.into(), 1.0.into(), (-1.0).into()],       // 2 positive inputs -> even
        [1.0.into(), 1.0.into(), 1.0.into()],          // 3 positive inputs -> odd
    ];

    let y_data = [
        (-1.0).into(), // even -> -1
        1.0.into(),    // odd -> 1
        1.0.into(),    // odd -> 1
        (-1.0).into(), // even -> -1
        1.0.into(),    // odd -> 1
        (-1.0).into(), // even -> -1
        (-1.0).into(), // even -> -1
        1.0.into(),    // odd -> 1
    ];

    let p_count = model.parameter_count();
    println!("parameters: {:?}", p_count);

    // Train the model
    let trainer = Trainer::new(model, GradientDescent::new(0.01), MseLoss {});
    let x_refs: Vec<&[AutogradValue]> = x_data.iter().map(|x| x.as_slice()).collect();
    trainer
        .train(&x_refs, &y_data, 10_000)
        .expect("Training failed");

    // Make predictions
    for x in x_refs {
        println!("Prediction: {:?}", trainer.predict_single(x));
    }

    Ok(())
}
