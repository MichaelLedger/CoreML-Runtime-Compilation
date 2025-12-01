import Foundation
import CoreML


public class GoogleNetPlacesInput : MLFeatureProvider {
    
    // Input image in the format of CVPixelBuffer
    public var sceneImage: CVPixelBuffer
    
    // Input feature name
    public var featureNames: Set<String> {
        get {
            return ["sceneImage"]
        }
    }
    
    // Value for a certain input feature.
    public func featureValue(for featureName: String) -> MLFeatureValue? {
        if (featureName == "sceneImage") {
            return MLFeatureValue(pixelBuffer: sceneImage)
        }
        return nil
    }
    
    public init(sceneImage: CVPixelBuffer) {
        self.sceneImage = sceneImage
    }
}


