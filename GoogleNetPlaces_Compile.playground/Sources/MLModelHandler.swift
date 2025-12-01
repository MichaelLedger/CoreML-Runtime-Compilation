import Foundation
import CoreML
import UIKit

public func buffer(from image: UIImage) -> CVPixelBuffer? {
    let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
    var pixelBuffer : CVPixelBuffer?
    let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(image.size.width), Int(image.size.height), kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
    guard (status == kCVReturnSuccess) else {
        return nil
    }
    
    CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
    let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)
    
    let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
    let context = CGContext(data: pixelData, width: Int(image.size.width), height: Int(image.size.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
    
    context?.translateBy(x: 0, y: image.size.height)
    context?.scaleBy(x: 1.0, y: -1.0)
    
    UIGraphicsPushContext(context!)
    image.draw(in: CGRect(x: 0, y: 0, width: image.size.width, height: image.size.height))
    UIGraphicsPopContext()
    CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
    
    return pixelBuffer
}



public class MLModelHandler {
    
    public init() {
        print("Initialized")
    }
    
    public func compileModel(path: URL) -> URL? {
        do {
            print("Compiling model at: \(path.path)")
            let compiledURL = try MLModel.compileModel(at: path)
            print("✓ Model compiled successfully to: \(compiledURL.path)")
            return compiledURL
        } catch let error as NSError {
            print("❌ Error in compiling model:")
            print("   Domain: \(error.domain)")
            print("   Code: \(error.code)")
            print("   Description: \(error.localizedDescription)")
            if let failureReason = error.localizedFailureReason {
                print("   Failure reason: \(failureReason)")
            }
            return nil
        }
    }
    
    
    public func compileModelAndPredict(path: URL, image: UIImage) {
        let compiledURL = compileModel(path: path)
        
        if let compiledURL = compiledURL {
            print("compiledURL:\(compiledURL)")
            
            do {
                // 1 - Load the compiled model
                print("Loading model from: \(compiledURL.path)")
                let model = try MLModel(contentsOf: compiledURL)
                print("✓ Model loaded successfully")
                
                // 2 - Resize image to 224x224 (required input size for GoogLeNet)
                UIGraphicsBeginImageContextWithOptions(CGSize(width: 224, height: 224), true, 2.0)
                image.draw(in: CGRect(x: 0, y: 0, width: 224, height: 224))
                let newImage = UIGraphicsGetImageFromCurrentImageContext()
                UIGraphicsEndImageContext()
                
                guard let resizedImage = newImage else {
                    print("Error: Failed to resize image")
                    return
                }
                
                // 3 - Convert image to CVPixelBuffer
                guard let imageBuffer = buffer(from: resizedImage) else {
                    print("Error: Failed to convert image to pixel buffer")
                    return
                }
                print("✓ Image converted to pixel buffer")
                
                // 4 - Make prediction
                print("Making prediction...")
                let result = try model.prediction(from: GoogleNetPlacesInput(sceneImage: imageBuffer))
                
                // 5 - Extract and display result
                if let label = result.featureValue(for: "sceneLabel")?.stringValue {
                    print("✓ Prediction result: \(label)")
                } else {
                    print("Warning: Could not extract scene label from prediction result")
                }
                
            } catch let error as NSError {
                print("❌ Error during model loading or prediction:")
                print("   Domain: \(error.domain)")
                print("   Code: \(error.code)")
                print("   Description: \(error.localizedDescription)")
                if let underlyingError = error.userInfo[NSUnderlyingErrorKey] as? NSError {
                    print("   Underlying error: \(underlyingError.localizedDescription)")
                }
            }
        }
    }

}

