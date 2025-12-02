import Foundation

public struct MDBestPhotoInferenceResult {
    let score: Float
    let label: String
}

public enum MDBestPhotoPredictorError: Swift.Error {
    case invalidModel
    case invalidInputTensor
    case invalidOutputTensor
}

public protocol MDBestPhotoPredictor {}

public extension MDBestPhotoPredictor {
    func topK(scores: [NSNumber], labels: [String], count: Int) -> [MDBestPhotoInferenceResult] {
        let zippedResults = zip(labels.indices, scores)
        let sortedResults = zippedResults.sorted { $0.1.floatValue > $1.1.floatValue }.prefix(count)
        return sortedResults.map { MDBestPhotoInferenceResult(score: $0.1.floatValue, label: labels[$0.0]) }
    }
}
