let img;
let model;
let modelPromise
let maxNumBoxes = 20
let minScore = 0.5

// original saved model from gdrive
const GOOGLE_CLOUD_STORAGE_DIR =
    'https://storage.googleapis.com/tfjs-models/savedmodel/';
const MODEL_URL =
    GOOGLE_CLOUD_STORAGE_DIR + 'coco-ssd-mobilenet_v1/model.json';
	

	
preload=()=>{
	console.log("kick start")
	modelPromise = load_model()
}

setup=()=>{
	predict()
}

// 
async function predict(){
	
	// retrieve image from html
	const img = document.querySelector("body > img")
	let tensorImage = tf.browser.fromPixels(img)
	let batch = await tf.expandDims(tensorImage)
	
	// note calling the load model async function only return a promise hense the await
	model = await modelPromise;
	const predictions = await model.executeAsync(batch)
	height  = batch.shape[1]
	width = batch.shape[2]
	
	// blocks the thead like await to extract value from tensor
	const scores = predictions[0].dataSync();
	const boxes = predictions[1].dataSync();
	
	// clear memory of unused tensor cause only python does it for you
    batch.dispose();
    tf.dispose(predictions);
	

	// limit prediction result based on top few with highest score aka probability
	const [maxScores, classes] = calculateMaxScores(scores, predictions[0].shape[1], predictions[0].shape[2]);
	
	// prune away boxes with high overlap so the image wont too messy
	// limit it to default 20 number of boxes like minScore is something we can toggle
	const indexTensor = tf.tidy(() => {
      const boxes2 =
          tf.tensor2d(boxes, [predictions[1].shape[1], predictions[1].shape[3]]);
      return tf.image.nonMaxSuppression(
          boxes2, maxScores, maxNumBoxes, minScore, minScore);
    });
	const indexes = indexTensor.dataSync()
    indexTensor.dispose();
	
	// format our prediction in to detection object
	let detections = buildDetectedObjects(width, height, boxes, maxScores, indexes, classes);
	
	
	// get canvas from html
	const c = document.getElementById('canvas');
	const context = c.getContext('2d');
	
	// draw our prediction with box and label
	context.drawImage(img, 0, 0,img.width,img.height);
	for (let i = 0; i < detections.length; i++) {
		context.beginPath();
		context.rect(...detections[i].bbox);
		context.lineWidth = 1;
		context.strokeStyle = 'green';
		context.fillStyle = 'green';
		context.stroke();
		context.fillText(
			detections[i].score.toFixed(3) + ' ' + detections[i].class, detections[i].bbox[0],
			detections[i].bbox[1] > 10 ? detections[i].bbox[1] - 5 : 10);
	}
}


async function load_model(){
	
	// localstorage instead
	//const modelUrl = 'https://storage.googleapis.com/tfjs-testing/tfjs-automl/object_detection/model.json';
	const modelUrl = 'http://localhost/efficientdet/cocossd2/model.json';
	
	const model = await tf.loadGraphModel(modelUrl)
	console.log('model loaded');
	return model
}

function calculateMaxScores(scores, numBoxes,numClasses) {
    const maxes = [];
    const classes = [];
    for (let i = 0; i < numBoxes; i++) {
      let max = Number.MIN_VALUE;
      let index = -1;
      for (let j = 0; j < numClasses; j++) {
        if (scores[i * numClasses + j] > max) {
          max = scores[i * numClasses + j];
          index = j;
        }
      }
      maxes[i] = max;
      classes[i] = index;
    }
    return [maxes, classes];
  }
  
 function buildDetectedObjects(width, height, boxes, scores, indexes, classes){
	 

    const count = indexes.length;
    const objects = [];
    for (let i = 0; i < count; i++) {
        const bbox = [];
        for (let j = 0; j < 4; j++) {
            bbox[j] = boxes[indexes[i] * 4 + j];
        }
        const minY = bbox[0] * height;
        const minX = bbox[1] * width;
        const maxY = bbox[2] * height;
        const maxX = bbox[3] * width;
        bbox[0] = minX;
        bbox[1] = minY;
        bbox[2] = maxX - minX;
        bbox[3] = maxY - minY;
        objects.push({
            bbox: bbox,
            class: CLASSES[classes[indexes[i]] + 1].displayName,
            score: scores[indexes[i]]
        });
    }
    return objects;
}