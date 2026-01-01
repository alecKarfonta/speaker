# YouTube API Setup Guide for Live Stream TTS

## Overview

This guide will walk you through setting up the YouTube Data API v3 to use with the Live Stream TTS feature. The integration allows you to connect to real YouTube live streams and automatically generate TTS for super chats.

## Prerequisites

- Google account
- YouTube live stream with super chat enabled
- Basic understanding of API keys

## Step 1: Create Google Cloud Project

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a project" at the top of the page
3. Click "New Project"
4. Enter a project name (e.g., "YouTube TTS Integration")
5. Click "Create"

## Step 2: Enable YouTube Data API v3

1. In the Google Cloud Console, make sure your new project is selected
2. Go to "APIs & Services" > "Library"
3. Search for "YouTube Data API v3"
4. Click on "YouTube Data API v3"
5. Click "Enable"

## Step 3: Create API Key

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "API Key"
3. Copy the generated API key
4. (Optional) Click "Restrict Key" to add restrictions:
   - **Application restrictions**: None (for browser use)
   - **API restrictions**: Select "YouTube Data API v3"

## Step 4: Configure API Key (Optional but Recommended)

For security, you can restrict your API key:

### HTTP Referrers (Recommended for production)
- Add your domain: `https://yourdomain.com/*`
- For local development: `http://localhost:*`

### API Restrictions
- Select "Restrict key"
- Choose "YouTube Data API v3"

## Step 5: Test Your Setup

1. Open the Live Stream TTS application at http://localhost:3002
2. Navigate to the "Live Stream TTS" tab
3. Click "Settings"
4. Enter your API key and a live stream video ID
5. Click "Connect to Stream"

## Finding a Live Stream Video ID

The video ID is the part after `v=` in a YouTube URL:
- URL: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`
- Video ID: `dQw4w9WgXcQ`

For live streams, make sure:
- The stream is currently live
- Super chat is enabled
- Live chat is not disabled

## API Quotas and Limits

### Default Quotas
- **Daily quota**: 10,000 units per day
- **Per-request cost**: 1-5 units depending on operation
- **Rate limit**: 100 requests per 100 seconds per user

### Request Costs
- Video details: 1 unit
- Live chat messages: 5 units per request
- Typical usage: ~300-500 units per hour of streaming

### Monitoring Usage
1. Go to "APIs & Services" > "Quotas"
2. Search for "YouTube Data API v3"
3. Monitor your daily usage

## Troubleshooting

### Common Errors

#### "Invalid API Key"
- Check that the API key is correct
- Ensure YouTube Data API v3 is enabled
- Verify API key restrictions allow your domain

#### "Video not found"
- Check the video ID is correct
- Ensure the video is a live stream
- Verify the stream is currently live

#### "Live chat not available"
- Check if live chat is enabled for the stream
- Some streams disable chat or super chat
- Private/unlisted streams may have restrictions

#### "Quota exceeded"
- You've hit the daily 10,000 unit limit
- Wait until quota resets (midnight Pacific Time)
- Consider requesting a quota increase

#### "Rate limit exceeded"
- The app is making requests too quickly
- The built-in rate limiting should handle this
- If persistent, check for multiple instances running

### Rate Limiting

The application includes built-in rate limiting:
- Minimum 1 second between requests
- Exponential backoff on errors
- Automatic retry with increasing delays
- Maximum 3 retries before giving up

### Connection Recovery

The app automatically handles:
- Network interruptions
- Temporary API errors
- Server-side rate limiting
- Stream disconnections

## Security Best Practices

### API Key Security
- Never commit API keys to version control
- Use environment variables in production
- Restrict API keys to specific domains
- Regularly rotate API keys

### Domain Restrictions
For production deployment:
```
https://yourdomain.com/*
https://*.yourdomain.com/*
```

For development:
```
http://localhost:*
http://127.0.0.1:*
```

## Cost Optimization

### Reducing API Usage
- The app automatically uses YouTube's recommended polling intervals
- Polling frequency adjusts based on stream activity
- Connection drops when stream ends
- Duplicate message filtering reduces unnecessary TTS calls

### Monitoring Costs
- Set up billing alerts in Google Cloud Console
- Monitor quota usage regularly
- Consider upgrading quota for high-volume usage

## Advanced Configuration

### Custom Polling Intervals
The app respects YouTube's `pollingIntervalMillis` response field, typically:
- Active streams: 5-10 seconds
- Inactive periods: 15-30 seconds
- Minimum enforced: 1 second

### Error Recovery Settings
Built-in settings (not user-configurable):
- Max consecutive errors: 5
- Backoff multiplier: 2x
- Maximum backoff delay: 30 seconds
- Connection timeout: 30 seconds

## Production Deployment

### Environment Variables
```bash
# Optional: Set default API key (not recommended for security)
YOUTUBE_API_KEY=your_api_key_here

# Better: Use runtime configuration
REACT_APP_YOUTUBE_API_ENABLED=true
```

### CORS Configuration
Ensure your backend allows requests from your frontend domain.

### HTTPS Requirements
YouTube API requires HTTPS in production. Use:
- SSL certificates
- Secure domain restrictions
- HTTPS redirects

## Support and Resources

### Official Documentation
- [YouTube Data API v3](https://developers.google.com/youtube/v3)
- [Live Chat API](https://developers.google.com/youtube/v3/live/docs)
- [Google Cloud Console](https://console.cloud.google.com/)

### Quota Increase Requests
If you need higher quotas:
1. Go to "APIs & Services" > "Quotas"
2. Select YouTube Data API v3
3. Click "Edit Quotas"
4. Request increase with justification

### Common Use Cases
- **Small streamers**: Default quota sufficient
- **Medium streamers**: May need quota increase
- **Large streamers**: Consider caching and optimization

## Testing Checklist

Before going live:
- [ ] API key works and is restricted
- [ ] Can connect to test live stream
- [ ] Super chats appear in real-time
- [ ] TTS generation works correctly
- [ ] Error handling works (test with invalid keys/IDs)
- [ ] Connection recovery works (test network interruption)
- [ ] Rate limiting prevents quota exhaustion
- [ ] Audio controls work properly

## Conclusion

The YouTube API integration provides a robust, production-ready solution for connecting TTS to live streams. The built-in error handling, rate limiting, and recovery mechanisms ensure reliable operation even with network issues or API limitations.

For additional support or questions about the integration, refer to the main README or check the application logs for detailed error information.
