 import UIKit
import CoreML
import PlaygroundSupport

let handler = MLModelHandler()
//URL of Apple's ml model
let fileURL = URL(string: "https://docs-assets.developer.apple.com/coreml/models/GoogLeNetPlaces.mlmodel")

//Destination url, in our case Playground Shared Data Directory
let documentsDirectory = playgroundSharedDataDirectory.appendingPathComponent("GoogleNetPlaces.mlmodel")

// Ensure the destination directory exists
let fileManager = FileManager.default
do {
    // Get the parent directory
    let parentDirectory = playgroundSharedDataDirectory
    print("Destination directory: \(parentDirectory.path)")
    
    // Create the directory if it doesn't exist
    if !fileManager.fileExists(atPath: parentDirectory.path) {
        try fileManager.createDirectory(at: parentDirectory, withIntermediateDirectories: true, attributes: nil)
        print("✓ Created destination directory")
    } else {
        print("✓ Destination directory already exists")
    }
} catch {
    print("Error creating destination directory: \(error)")
}

// Function to run prediction with the model
func runPrediction() {
    let imageName = "christmas.jpg" //christmas.jpg //forest.jpg
    if let image = UIImage(named: imageName) {
        handler.compileModelAndPredict(path: documentsDirectory, image: image)
    } else {
        print("Warning: Could not load \(imageName) image")
    }
}

// Check if model file already exists
if fileManager.fileExists(atPath: documentsDirectory.path) {
    print("✓ Model file already exists, skipping download")
    
    // Verify file size to ensure it's valid
    do {
        let attributes = try fileManager.attributesOfItem(atPath: documentsDirectory.path)
        if let fileSize = attributes[.size] as? Int64 {
            print("Existing file size: \(fileSize) bytes")
            
            if fileSize > 0 {
                // File exists and is not empty, use it directly
                runPrediction()
            } else {
                print("File exists but is empty, will re-download")
                try fileManager.removeItem(at: documentsDirectory)
            }
        }
    } catch {
        print("Error checking existing file: \(error)")
    }
} else {
    print("Model file not found, starting download...")
    
    let task = URLSession.shared.downloadTask(with: fileURL!) { localURL, urlResponse, error in
        // Check for download errors first
        if let error = error {
            print("Download failed with error: \(error.localizedDescription)")
            return
        }
        
        // Check HTTP response status
        if let httpResponse = urlResponse as? HTTPURLResponse {
            print("HTTP Status Code: \(httpResponse.statusCode)")
            if httpResponse.statusCode != 200 {
                print("Server returned non-200 status code")
                return
            }
        }
        
        guard let localURL = localURL else {
            print("No local URL provided")
            return
        }
        
        print("Download completed. Temporary file at: \(localURL.path)")
        
        do {
            // Verify temporary file exists before copying
            if !fileManager.fileExists(atPath: localURL.path) {
                print("ERROR: Temporary file does not exist at \(localURL.path)")
                return
            }
            
            print("Copying from \(localURL.path) to \(documentsDirectory.path)")
            
            // Copy from temporary location to custom location
            try fileManager.copyItem(at: localURL, to: documentsDirectory)
            print("✓ Model downloaded and saved successfully")
            
            // Verify the file was copied
            if fileManager.fileExists(atPath: documentsDirectory.path) {
                let attributes = try fileManager.attributesOfItem(atPath: documentsDirectory.path)
                if let fileSize = attributes[.size] as? Int64 {
                    print("Downloaded file size: \(fileSize) bytes")
                }
            }
            
            // Now use the model for prediction
            runPrediction()
                        
        } catch {
            print("Error in copying to playground's documents directory: \(error)")
            print("Error details: \(error.localizedDescription)")
        }
    }
    
    task.resume()
}
