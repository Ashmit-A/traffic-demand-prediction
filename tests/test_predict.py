from src.predict import predict_single


def test_prediction():

    sample = {
        "geohash": "qp03w1",
        "day": 48,
        "timestamp": "5:0",
        "RoadType": "Residential",
        "NumberofLanes": 3,
        "LargeVehicles": "Allowed",
        "Landmarks": "Yes",
        "Temperature": 11.757823186682774,
        "Weather": "Rainy"
    }

    prediction = predict_single(sample)

    assert isinstance(prediction, float)