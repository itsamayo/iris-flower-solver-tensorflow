const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
var iris = require("./iris.json")
var irisTesting = require("./iris-testing.json")

// Convert/setup our data
const trainingData = tf.tensor2d(iris.map(item => [
    item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
]))
const ouputData = tf.tensor2d(iris.map(item => [
    item.species === "setosa" ? 1 : 0,
    item.species === "virginica" ? 1 : 0,
    item.species === "versicolor" ? 1 : 0,
]))
const testingData = tf.tensor2d(irisTesting.map(item => [
    item.sepal_length, item.sepal_width, item.petal_length, item.petal_width
]))

// Build nearal network
const model = tf.sequential()

// NOTE Sigmoid is good for classification  where you have multiple output dimensions that are ideally going to be 1 or 0
// NOTE regression is best for single output dimensions like a score from 0 - 100
model.add(tf.layers.dense({
    inputShape: [4],
    activation: "sigmoid",
    units: 5,
}))
model.add(tf.layers.dense({
    inputShape: [5],
    activation: "sigmoid",
    units: 3,
}))
model.add(tf.layers.dense({
    activation: "sigmoid",
    units: 3,
}))
model.compile({
    loss: "meanSquaredError",
    optimizer: tf.train.adam(.06)
})

// Train/fit our network
const startTime = Date.now()
model.fit(trainingData, ouputData, {epochs: 100})
    .then((history) => {
        console.log(history)
        console.log("DONE!", Date.now() - startTime)
        model.predict(testingData).print()
    })

// Test network
