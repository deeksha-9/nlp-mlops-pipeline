# tests/test_api.py
"""
Test suite for the Sentiment Analysis API
Tests all endpoints to ensure they work correctly before deployment.
"""

import pytest
from unittest.mock import Mock
from fastapi.testclient import TestClient
import api.main
from api.main import app

# ============================================================
# SETUP: Create mock model for testing
# ============================================================

def create_mock_model():
    """Create a mock sentiment model that mimics the real model"""
    mock = Mock()
    
    def predict_side_effect(texts):
        """Mock prediction logic - keyword matching"""
        results = []
        positive_words = ["good", "great", "amazing", "loved", "excellent", "superb", "engaging", "best", "wonderful", "fantastic"]
        negative_words = ["bad", "terrible", "worst", "hate", "awful", "boring", "waste", "horrible"]
        
        text_list = texts if isinstance(texts, list) else [texts]
        for text in text_list:
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            # 1 for positive, 0 for negative
            prediction = 1 if pos_count > neg_count else 0
            results.append(prediction)
        
        return results[0] if not isinstance(texts, list) else results
    
    def predict_proba_side_effect(texts):
        """Mock probability predictions"""
        results = []
        positive_words = ["good", "great", "amazing", "loved", "excellent", "superb", "engaging", "best", "wonderful", "fantastic"]
        negative_words = ["bad", "terrible", "worst", "hate", "awful", "boring", "waste", "horrible"]
        
        text_list = texts if isinstance(texts, list) else [texts]
        for text in text_list:
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            total = pos_count + neg_count + 2  # +2 to avoid division by zero
            neg_prob = neg_count / total
            pos_prob = pos_count / total
            
            results.append([neg_prob, pos_prob])
        
        return results[0] if not isinstance(texts, list) else results
    
    mock.predict = Mock(side_effect=predict_side_effect)
    mock.predict_proba = Mock(side_effect=predict_proba_side_effect)
    return mock

# ============================================================
# SETUP: Initialize mock model before creating test client
# ============================================================

# Set the mock model in api.main so it's available during tests
api.main.model = create_mock_model()

# TestClient simulates HTTP requests without starting a real server
# It's faster than real HTTP calls and works offline
client = TestClient(app)

# ============================================================
# TEST 1: Root endpoint
# ============================================================

def test_root_endpoint():
    """
    Test that the root endpoint (/) returns API information.
    
    What it checks:
    - Status code is 200 (success)
    - Response contains expected fields
    """
    response = client.get("/")
    
    # Check status code
    assert response.status_code == 200, "Root endpoint should return 200"
    
    # Check response structure
    data = response.json()
    assert "message" in data, "Response should contain 'message' field"
    assert "docs" in data, "Response should contain 'docs' field"
    assert "version" in data, "Response should contain 'version' field"
    
    # Check specific values
    assert data["docs"] == "/docs", "Docs URL should be /docs"
    
    print("✓ Root endpoint test passed")

# ============================================================
# TEST 2: Health check endpoint
# ============================================================

def test_health_endpoint():
    """
    Test that the /health endpoint returns server status.
    
    What it checks:
    - Status code is 200
    - Model is loaded
    - Response has correct structure
    """
    response = client.get("/health")
    
    # Check status code
    assert response.status_code == 200, "Health endpoint should return 200"
    
    # Check response structure
    data = response.json()
    assert "status" in data, "Response should contain 'status'"
    assert "model_loaded" in data, "Response should contain 'model_loaded'"
    assert "model_name" in data, "Response should contain 'model_name'"
    
    # Check model is loaded
    assert data["model_loaded"] == True, "Model should be loaded"
    assert data["status"] == "healthy", "Status should be healthy"
    assert data["model_name"] == "sentiment-model", "Model name should match"
    
    print("✓ Health endpoint test passed")

# ============================================================
# TEST 3: Predict endpoint with positive review
# ============================================================

def test_predict_positive_sentiment():
    """
    Test prediction with a clearly positive review.
    
    What it checks:
    - Request succeeds (200)
    - Sentiment is 'positive'
    - Confidence is a valid number between 0 and 1
    - Response structure is correct
    """
    # Prepare test data
    test_input = {
        "text": "This movie was absolutely amazing! I loved every minute of it. The acting was superb and the plot was engaging."
    }
    
    # Make POST request to /predict
    response = client.post("/predict", json=test_input)
    
    # Check status code
    assert response.status_code == 200, "Predict endpoint should return 200"
    
    # Parse response
    data = response.json()
    
    # Check response structure
    assert "text" in data, "Response should contain 'text'"
    assert "sentiment" in data, "Response should contain 'sentiment'"
    assert "confidence" in data, "Response should contain 'confidence'"
    assert "model_version" in data, "Response should contain 'model_version'"
    
    # Check sentiment is correct
    assert data["sentiment"] == "positive", f"Sentiment should be positive, got {data['sentiment']}"
    
    # Check confidence is valid
    assert isinstance(data["confidence"], float), "Confidence should be a float"
    assert 0 <= data["confidence"] <= 1, f"Confidence should be between 0 and 1, got {data['confidence']}"
    
    # Check text matches input
    assert data["text"] == test_input["text"], "Returned text should match input"
    
    print(f"✓ Positive sentiment test passed (confidence: {data['confidence']:.3f})")

# ============================================================
# TEST 4: Predict endpoint with negative review
# ============================================================

def test_predict_negative_sentiment():
    """
    Test prediction with a clearly negative review.
    
    What it checks:
    - Request succeeds
    - Sentiment is 'negative'
    - Confidence is valid
    """
    test_input = {
        "text": "This was the worst movie I've ever seen. Terrible acting, boring plot, complete waste of time."
    }
    
    response = client.post("/predict", json=test_input)
    
    assert response.status_code == 200
    
    data = response.json()
    
    # Check sentiment
    assert data["sentiment"] == "negative", f"Sentiment should be negative, got {data['sentiment']}"
    
    # Check confidence
    assert 0 <= data["confidence"] <= 1
    
    print(f"✓ Negative sentiment test passed (confidence: {data['confidence']:.3f})")

# ============================================================
# TEST 5: Edge case - empty text
# ============================================================

def test_predict_empty_text():
    """
    Test that empty text is rejected.
    
    What it checks:
    - Returns 422 (Pydantic validation error)
    - FastAPI uses 422 for request validation failures
    """
    test_input = {
        "text": ""
    }
    
    response = client.post("/predict", json=test_input)
    
    # Should return 422 for Pydantic validation error (min_length constraint)
    assert response.status_code == 422, "Empty text should return 422 (validation error)"
    
    data = response.json()
    assert "detail" in data, "Error response should contain 'detail'"
    
    print("✓ Empty text rejection test passed")

# ============================================================
# TEST 6: Edge case - very short text
# ============================================================

def test_predict_short_text():
    """
    Test prediction with very short input.
    
    What it checks:
    - Model can handle short input
    - Returns valid sentiment
    """
    test_input = {
        "text": "Great!"
    }
    
    response = client.post("/predict", json=test_input)
    
    assert response.status_code == 200
    
    data = response.json()
    assert data["sentiment"] in ["positive", "negative"]
    assert 0 <= data["confidence"] <= 1
    
    print(f"✓ Short text test passed (sentiment: {data['sentiment']})")

# ============================================================
# TEST 7: Edge case - long text
# ============================================================

def test_predict_long_text():
    """
    Test prediction with long input.
    
    What it checks:
    - Model can handle long reviews
    - Returns valid result
    """
    # Create a long review (repeated text)
    long_review = "This movie was amazing! " * 100  # 2400+ characters
    
    test_input = {
        "text": long_review
    }
    
    response = client.post("/predict", json=test_input)
    
    assert response.status_code == 200
    
    data = response.json()
    assert data["sentiment"] in ["positive", "negative"]
    assert 0 <= data["confidence"] <= 1
    
    print(f"✓ Long text test passed ({len(long_review)} chars)")

# ============================================================
# TEST 8: Edge case - text with special characters
# ============================================================

def test_predict_special_characters():
    """
    Test that special characters don't break the API.
    
    What it checks:
    - API handles emojis, punctuation, etc.
    - Returns valid prediction
    """
    test_input = {
        "text": "Amazing!!! 😍🎬 Best movie ever!!! 10/10 ⭐⭐⭐⭐⭐"
    }
    
    response = client.post("/predict", json=test_input)
    
    assert response.status_code == 200
    
    data = response.json()
    assert data["sentiment"] in ["positive", "negative"]
    
    print(f"✓ Special characters test passed (sentiment: {data['sentiment']})")

# ============================================================
# TEST 9: Invalid request - missing 'text' field
# ============================================================

def test_predict_missing_text_field():
    """
    Test that requests without 'text' field are rejected.
    
    What it checks:
    - Returns 422 (Unprocessable Entity)
    - Pydantic validation works
    """
    # Send request without 'text' field
    test_input = {
        "review": "This should fail"  # Wrong field name
    }
    
    response = client.post("/predict", json=test_input)
    
    # Should return 422 for validation error
    assert response.status_code == 422, "Missing required field should return 422"
    
    print("✓ Missing field validation test passed")

# ============================================================
# TEST 10: Batch prediction endpoint
# ============================================================

def test_batch_prediction():
    """
    Test the /predict/batch endpoint.
    
    What it checks:
    - Can process multiple texts at once
    - All predictions are valid
    - Returns correct count
    """
    test_texts = [
        "Amazing movie, loved it!",
        "Terrible film, waste of time.",
        "It was okay, nothing special."
    ]
    
    response = client.post("/predict/batch", json=test_texts)
    
    assert response.status_code == 200
    
    data = response.json()
    
    # Check response structure
    assert "predictions" in data
    assert "count" in data
    
    # Check count matches
    assert data["count"] == len(test_texts)
    assert len(data["predictions"]) == len(test_texts)
    
    # Check each prediction is valid
    for i, prediction in enumerate(data["predictions"]):
        assert "text" in prediction
        assert "sentiment" in prediction
        assert "confidence" in prediction
        
        assert prediction["sentiment"] in ["positive", "negative"]
        assert 0 <= prediction["confidence"] <= 1
        assert prediction["text"] == test_texts[i]
    
    print(f"✓ Batch prediction test passed ({data['count']} texts)")

# ============================================================
# TEST 11: Batch prediction - empty list
# ============================================================

def test_batch_prediction_empty():
    """
    Test batch endpoint rejects empty list.
    """
    response = client.post("/predict/batch", json=[])
    
    assert response.status_code == 400
    
    print("✓ Empty batch rejection test passed")

# ============================================================
# TEST 12: Batch prediction - too many texts
# ============================================================

def test_batch_prediction_too_many():
    """
    Test batch endpoint rejects more than 100 texts.
    """
    # Create 101 texts
    too_many_texts = ["Test text"] * 101
    
    response = client.post("/predict/batch", json=too_many_texts)
    
    assert response.status_code == 400
    
    data = response.json()
    assert "Max 100" in data["detail"]
    
    print("✓ Batch size limit test passed")

# ============================================================
# OPTIONAL: Pytest configuration
# ============================================================

# This runs before all tests (optional)
@pytest.fixture(scope="module")
def setup_teardown():
    """
    Setup: Runs once before all tests
    Teardown: Runs once after all tests
    """
    print("\n" + "="*60)
    print("STARTING API TESTS")
    print("="*60)
    
    yield  # Tests run here
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)

# Use the fixture in a test
def test_setup_works(setup_teardown):
    """Dummy test to trigger setup/teardown"""
    assert True