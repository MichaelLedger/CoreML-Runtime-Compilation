 import UIKit
import CoreML
import PlaygroundSupport

let handler = MLModelHandler()

// Construct the absolute path to the model in the playground's Resources folder
// Since the playground is at: /Users/gavinxiang/Downloads/CoreML-Runtime-Compilation/GoogleNetPlaces_Compile.playground/
let modelPath = "/Users/gavinxiang/Downloads/CoreML-Runtime-Compilation/GoogleNetPlaces_Compile.playground/Resources/GoogLeNetPlaces.mlmodel"
let modelURL = URL(fileURLWithPath: modelPath)

print("Looking for model at: \(modelURL.path)")

// Verify the model file exists
let fileManager = FileManager.default
if !fileManager.fileExists(atPath: modelURL.path) {
    print("❌ Error: Model file does not exist at: \(modelURL.path)")
    
    // Let's check what's in the Resources folder
    let resourcesPath = "/Users/gavinxiang/Downloads/CoreML-Runtime-Compilation/GoogleNetPlaces_Compile.playground/Resources"
    
    print("\nChecking Resources folder at: \(resourcesPath)")
    if fileManager.fileExists(atPath: resourcesPath) {
        do {
            let contents = try fileManager.contentsOfDirectory(atPath: resourcesPath)
            print("Files in Resources folder: \(contents)")
        } catch {
            print("Could not list Resources folder: \(error)")
        }
    } else {
        print("Resources folder doesn't exist at: \(resourcesPath)")
    }
} else {
    print("✓ Model file verified at: \(modelURL.path)")
    
    // Function to run prediction with the model
    func runPrediction() {
        let imageName = "forest.jpg"
        if let image = UIImage(named: imageName) {
            print("✓ Image loaded: \(imageName)")
            handler.compileModelAndPredict(path: modelURL, image: image)
        } else {
            print("❌ Warning: Could not load \(imageName) image from Resources")
        }
    }
    
    // Run prediction with local model
    runPrediction()
}
