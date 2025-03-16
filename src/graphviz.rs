use crate::autograd::{AutogradValue, Module, Operation, ValueData};
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;

pub fn dot_to_file_with_model<M: Module>(
    value: &AutogradValue,
    model: &M,
    inputs: &[&[AutogradValue]],
) {
    let mut dot = String::from("digraph G {\n");
    let mut visited: HashSet<*const ValueData> = HashSet::new();

    // Get all parameters directly from the model
    let params: HashSet<*const ValueData> = model
        .parameters()
        .iter()
        .map(|p| p.0.as_ptr() as *const ValueData)
        .collect();

    let inputs: HashSet<*const ValueData> = inputs
        .iter()
        .flat_map(|&p| p)
        .map(|p| p.0.as_ptr() as *const ValueData)
        .collect();

    generate_dot(value, &mut dot, &mut visited, &params, &inputs);
    dot.push_str("}\n");

    std::fs::create_dir_all("./graphviz").expect("Unable to create graphviz directory");
    let mut file = File::create("./graphviz/graph.dot").expect("Unable to create file");
    file.write_all(dot.as_bytes())
        .expect("Unable to write data");
    file.flush().expect("Unable to flush file");
}

fn generate_dot(
    value: &AutogradValue,
    dot: &mut String,
    visited: &mut HashSet<*const ValueData>,
    params: &HashSet<*const ValueData>,
    inputs: &HashSet<*const ValueData>,
) {
    // NOTE: Using pointer address as unique identifier works for this educational implementation
    // but could be problematic in production code if memory is reallocated.
    let ptr = value.0.as_ptr() as *const ValueData;

    // If already visited, return to avoid duplication.
    if !visited.insert(ptr) {
        return;
    }

    let label = format!("data: {:.2}, grad: {:.2}", value.data(), value.grad());

    if params.contains(&ptr) {
        dot.push_str(&format!("  node{:p} [label=\"{}\", color=darkgreen, style=filled, fillcolor=lightgreen, shape=box];\n",
                              ptr, label));
    } else if inputs.contains(&ptr) {
        dot.push_str(&format!("  node{:p} [label=\"{}\", color=darkblue, style=filled, fillcolor=lightblue, shape=box];\n",
                              ptr, label));
    } else {
        dot.push_str(&format!("  node{:p} [label=\"{}\"];\n", ptr, label));
    }

    // Depending on the operation, traverse its children
    let binding = value.0.borrow();
    let op_clone = binding.op();
    match op_clone {
        Operation::None => {} // Leaf node, no children.
        Operation::Add(a, b) => {
            dot.push_str(&format!(
                "  node{:p} -> node{:p} [label=\"add\"];\n",
                ptr,
                a.0.as_ptr()
            ));
            dot.push_str(&format!(
                "  node{:p} -> node{:p} [label=\"add\"];\n",
                ptr,
                b.0.as_ptr()
            ));
            generate_dot(a, dot, visited, params, inputs);
            generate_dot(b, dot, visited, params, inputs);
        }
        Operation::Multiply(a, b) => {
            dot.push_str(&format!(
                "  node{:p} -> node{:p} [label=\"mul\"];\n",
                ptr,
                a.0.as_ptr()
            ));
            dot.push_str(&format!(
                "  node{:p} -> node{:p} [label=\"mul\"];\n",
                ptr,
                b.0.as_ptr()
            ));
            generate_dot(a, dot, visited, params, inputs);
            generate_dot(b, dot, visited, params, inputs);
        }
        Operation::Power(a, n) => {
            dot.push_str(&format!(
                "  node{:p} -> node{:p} [label=\"pow {:.2}\"];\n",
                ptr,
                a.0.as_ptr(),
                n
            ));
            generate_dot(a, dot, visited, params, inputs);
        }
        Operation::Relu(a) => {
            dot.push_str(&format!(
                "  node{:p} -> node{:p} [label=\"relu\"];\n",
                ptr,
                a.0.as_ptr()
            ));
            generate_dot(a, dot, visited, params, inputs);
        }
        Operation::Negate(a) => {
            dot.push_str(&format!(
                "  node{:p} -> node{:p} [label=\"neg\"];\n",
                ptr,
                a.0.as_ptr()
            ));
            generate_dot(a, dot, visited, params, inputs);
        }
        Operation::Exp(a) => {
            dot.push_str(&format!(
                "  node{:p} -> node{:p} [label=\"exp\"];\n",
                ptr,
                a.0.as_ptr()
            ));
            generate_dot(a, dot, visited, params, inputs);
        }
    }
}
