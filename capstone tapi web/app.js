const tf = require('@tensorflow/tfjs');
const tfnode = require('@tensorflow/tfjs-node');

const fs = require('fs');

var priceDict = {
    0: "Apple_Bad", 1: "Banana_Bad", 2: "Guava_Bad",3: "Lime_Bad", 4: "Orange_Bad", 5: "Pomegranate_Bad",
    6: "Apple_Good", 7: "Banana_Good", 8: "Guava_Good",9: "Lime_Good", 10: "Orange_Good", 11: "Pomegranate_Good",
    12: "Apple_Mixed", 13: "Banana_Mixed", 14: "Guava_Mixed",15: "Lime_Mixed", 16: "Orange_Mixed", 17: "Pomegranate_Mixed"


  };

async function loadModel(path) {
// Get file path
    const handler = tfnode.io.fileSystem(path);
  
    // Loda model from file path
    const model = await tf.loadLayersModel(handler);
  
    console.log("Model loaded!");
    return model
}

async function loadImage(path) {
    // Read imgae as file buffer
    const imageBuffer = fs.readFileSync(path);
  
    // Convert buffer imgage into tensor
    const tensor = tfnode.node.decodeImage(imageBuffer);
  
    // Resize image to fit the model
    const resizedTensor  = tf.image.resizeBilinear(tensor, [64, 64]).div(tf.scalar(255));
    const expandedTensor = resizedTensor.expandDims(0);
  
    console.log("Image loaded!");
    return expandedTensor;
  }

  async function predict() {
    // Load model
    const modelClass = await loadModel('D:/Bangkit/capstone tapi web/tfjs_model-coba/model.json');
  
    // load image
    const img = await loadImage('D:/Bangkit/capstone tapi web/IMG_20190822_080144.jpg');
  
    // Doing prediction
    const predictClass =  modelClass.predict(img);
  
    // quality prediction
    objectClass = predictClass.argMax(1).arraySync()[0];
    classResult = priceDict[objectClass];
    console.log(classResult);
    console.log(objectClass)
    
    

  }
  
  predict();