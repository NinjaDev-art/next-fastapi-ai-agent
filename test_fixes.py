#!/usr/bin/env python3
"""
Test script to verify the fixes for PDF processing and image handling.
"""

import asyncio
import logging
from app.services.chat_service import chat_service
from app.utils.file_processor import file_processor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_pdf_processing():
    """Test PDF file processing doesn't return dict objects"""
    print("Testing PDF processing...")
    
    # Test with a mock PDF file URL (this would normally be a real file)
    try:
        # This should now handle errors gracefully and return a string
        result = file_processor.process_files(["test.pdf"])
        print(f"PDF processing result type: {type(result)}")
        print(f"PDF processing success: {isinstance(result, str)}")
        return isinstance(result, str)
    except Exception as e:
        print(f"PDF processing error: {e}")
        return False

def test_image_identification():
    """Test image file identification"""
    print("Testing image file identification...")
    
    test_files = [
        "document.pdf",
        "image1.png", 
        "image2.jpg",
        "spreadsheet.xlsx",
        "photo.jpeg",
        "icon.gif"
    ]
    
    image_files, text_files = file_processor.identify_files(test_files)
    
    print(f"Image files: {image_files}")
    print(f"Text files: {text_files}")
    
    expected_images = ["image1.png", "image2.jpg", "photo.jpeg", "icon.gif"]
    expected_texts = ["document.pdf", "spreadsheet.xlsx"]
    
    images_correct = set(image_files) == set(expected_images)
    texts_correct = set(text_files) == set(expected_texts)
    
    print(f"Image identification correct: {images_correct}")
    print(f"Text identification correct: {texts_correct}")
    
    return images_correct and texts_correct

def test_multimodal_message_creation():
    """Test multimodal message creation"""
    print("Testing multimodal message creation...")
    
    try:
        # Test OpenAI multimodal message
        openai_msg = chat_service.create_multimodal_message(
            "Describe this image",
            ["https://example.com/image.jpg"],
            "openai"
        )
        
        print(f"OpenAI multimodal message: {openai_msg}")
        
        # Test Anthropic multimodal message  
        anthropic_msg = chat_service.create_multimodal_message(
            "Analyze this image",
            ["https://example.com/image.png"],
            "anthropic"
        )
        
        print(f"Anthropic multimodal message structure: {type(anthropic_msg)}")
        
        # Check structure
        openai_valid = (
            openai_msg.get("role") == "user" and
            isinstance(openai_msg.get("content"), list) and
            len(openai_msg["content"]) >= 2
        )
        
        anthropic_valid = (
            anthropic_msg.get("role") == "user" and
            isinstance(anthropic_msg.get("content"), list)
        )
        
        print(f"OpenAI message structure valid: {openai_valid}")
        print(f"Anthropic message structure valid: {anthropic_valid}")
        
        return openai_valid and anthropic_valid
        
    except Exception as e:
        print(f"Multimodal message creation error: {e}")
        return False

async def main():
    """Run all tests"""
    print("=" * 50)
    print("Running fix verification tests")
    print("=" * 50)
    
    # Test 1: PDF processing
    pdf_test = await test_pdf_processing()
    
    # Test 2: Image identification
    image_test = test_image_identification()
    
    # Test 3: Multimodal message creation
    multimodal_test = test_multimodal_message_creation()
    
    print("=" * 50)
    print("Test Results:")
    print(f"PDF Processing: {'✓ PASS' if pdf_test else '✗ FAIL'}")
    print(f"Image Identification: {'✓ PASS' if image_test else '✗ FAIL'}")
    print(f"Multimodal Messages: {'✓ PASS' if multimodal_test else '✗ FAIL'}")
    
    all_passed = pdf_test and image_test and multimodal_test
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("=" * 50)
    
    return all_passed

if __name__ == "__main__":
    asyncio.run(main()) 