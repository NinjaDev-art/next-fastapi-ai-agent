# Image Input Support for FastAPI Chatbot

This chatbot supports using images as input prompts with vision-capable AI models. Here's how to implement and use this feature.

## Supported Models & Providers

### OpenAI
- **Models**: GPT-4V, GPT-4o, GPT-4o-mini (with vision)
- **Format**: Direct URL or base64
- **Token Cost**: 85-170 tokens per image

### Anthropic
- **Models**: Claude 3 (Haiku, Sonnet, Opus)
- **Format**: Base64 encoded
- **Token Cost**: ~170 tokens per image

### Google
- **Models**: Gemini Pro Vision, Gemini 1.5
- **Format**: Base64 encoded
- **Token Cost**: ~170 tokens per image

## Supported Image Formats

- `.png`
- `.jpg` / `.jpeg`
- `.gif`
- `.bmp`
- `.tiff`
- `.ico`
- `.webp`

## How to Use Images as Input

### 1. API Request Format

```json
{
  "prompt": "What do you see in this image?",
  "files": [
    "https://your-cdn.com/image1.jpg",
    "uploads/image2.png"
  ],
  "model": "gpt-4o",
  "email": "user@example.com",
  "sessionId": "session123",
  "chatHistory": [],
  "reGenerate": false,
  "chatType": 0
}
```

### 2. Frontend Implementation Example

```javascript
// Upload image and send to chat
const sendImagePrompt = async (imageFile, textPrompt) => {
  // First upload the image
  const formData = new FormData();
  formData.append('file', imageFile);
  
  const uploadResponse = await fetch('/api/upload', {
    method: 'POST',
    body: formData
  });
  
  const { imageUrl } = await uploadResponse.json();
  
  // Then send chat request with image
  const chatResponse = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      prompt: textPrompt,
      files: [imageUrl], // Include the uploaded image URL
      model: 'gpt-4o',
      email: 'user@example.com',
      sessionId: generateSessionId(),
      chatHistory: [],
      reGenerate: false,
      chatType: 0
    })
  });
};
```

### 3. React Component Example

```jsx
import React, { useState } from 'react';

const ImageChatComponent = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
    }
  };

  const sendMessage = async () => {
    if (!selectedImage || !prompt) return;

    // Upload image first
    const formData = new FormData();
    formData.append('file', selectedImage);
    
    const uploadRes = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });
    
    const { imageUrl } = await uploadRes.json();

    // Send chat request
    const chatRes = await fetch('/api/chat/generateText', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        prompt: prompt,
        files: [imageUrl],
        model: 'gpt-4o',
        email: 'user@example.com',
        sessionId: 'session123',
        chatHistory: [],
        reGenerate: false,
        chatType: 0
      })
    });

    const result = await chatRes.json();
    setResponse(result.data);
  };

  return (
    <div>
      <input 
        type="file" 
        accept="image/*" 
        onChange={handleImageUpload}
      />
      
      {selectedImage && (
        <img 
          src={URL.createObjectURL(selectedImage)} 
          alt="Preview" 
          style={{ maxWidth: '300px', maxHeight: '300px' }}
        />
      )}
      
      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="Ask something about the image..."
      />
      
      <button onClick={sendMessage}>
        Send Message
      </button>
      
      {response && (
        <div>
          <h3>AI Response:</h3>
          <p>{response}</p>
        </div>
      )}
    </div>
  );
};

export default ImageChatComponent;
```

## Advanced Usage

### Multiple Images
You can send multiple images in a single request:

```json
{
  "prompt": "Compare these two images",
  "files": [
    "image1.jpg",
    "image2.png"
  ],
  "model": "gpt-4o"
}
```

### Combining with RAG
Images can be combined with document processing:

```json
{
  "prompt": "Analyze this chart and compare it with the data in the document",
  "files": [
    "chart.png",
    "document.pdf"
  ],
  "model": "gpt-4o"
}
```

### Image Detail Levels (OpenAI)
For OpenAI models, you can control the detail level programmatically by modifying the `format_image_content` method:

```python
# In the format_image_content method, change:
"detail": "high"  # Options: "low", "high", "auto"
```

## Best Practices

1. **Image Quality**: Higher resolution images provide better analysis but cost more tokens
2. **File Size**: Keep images under 20MB for optimal performance
3. **Context**: Provide clear text prompts describing what you want to know about the image
4. **Model Selection**: Use vision-capable models (GPT-4V, Claude 3, Gemini Vision)

## Error Handling

The system will automatically:
- Detect image files by extension
- Convert images to the appropriate format for each provider
- Handle unsupported formats gracefully
- Provide fallback behavior for non-vision models

## Getting Image Support Information

Use the info endpoint to get current capabilities:

```javascript
const getImageInfo = async () => {
  const response = await fetch('/api/chat/image-support-info');
  const info = await response.json();
  console.log(info.data);
};
```

This will return supported formats, token costs, and usage guidelines for each provider. 