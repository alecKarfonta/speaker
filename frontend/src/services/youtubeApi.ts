// YouTube Data API service for browser-based integration
// Based on the logic from YoutubeComments.ipynb notebook

export interface SuperChatDetails {
  amountDisplayString: string;
  amountMicros: string;
  currency: string;
  tier: number;
  userComment?: string;
}

export interface AuthorDetails {
  channelId: string;
  channelUrl: string;
  displayName: string;
  profileImageUrl: string;
  isVerified: boolean;
  isChatOwner: boolean;
  isChatSponsor: boolean;
  isChatModerator: boolean;
}

export interface LiveChatMessage {
  kind: string;
  etag: string;
  id: string;
  snippet: {
    type: string;
    liveChatId: string;
    authorChannelId: string;
    publishedAt: string;
    hasDisplayContent: boolean;
    displayMessage: string;
    textMessageDetails?: {
      messageText: string;
    };
    superChatDetails?: SuperChatDetails;
    fanFundingEventDetails?: {
      amountDisplayString: string;
      amountMicros: string;
      currency: string;
      userComment?: string;
    };
  };
  authorDetails: AuthorDetails;
}

export interface LiveChatResponse {
  kind: string;
  etag: string;
  nextPageToken?: string;
  pollingIntervalMillis: number;
  pageInfo: {
    totalResults: number;
    resultsPerPage: number;
  };
  items: LiveChatMessage[];
}

export interface VideoResponse {
  kind: string;
  etag: string;
  items: Array<{
    id: string;
    liveStreamingDetails?: {
      actualStartTime?: string;
      actualEndTime?: string;
      scheduledStartTime?: string;
      concurrentViewers?: string;
      activeLiveChatId?: string;
    };
  }>;
}

export class YouTubeApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public code?: string,
    public details?: any
  ) {
    super(message);
    this.name = 'YouTubeApiError';
  }
}

export class YouTubeApiService {
  private apiKey: string;
  private baseUrl = 'https://www.googleapis.com/youtube/v3';
  private rateLimitDelay = 1000; // 1 second between requests
  private lastRequestTime = 0;
  private maxRetries = 3;
  private backoffMultiplier = 2;

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  private async makeRequest<T>(url: string, retryCount = 0): Promise<T> {
    // Rate limiting: ensure minimum delay between requests
    const now = Date.now();
    const timeSinceLastRequest = now - this.lastRequestTime;
    if (timeSinceLastRequest < this.rateLimitDelay) {
      await new Promise(resolve => 
        setTimeout(resolve, this.rateLimitDelay - timeSinceLastRequest)
      );
    }
    this.lastRequestTime = Date.now();

    try {
      const response = await fetch(url);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        
        // Handle specific YouTube API errors
        if (response.status === 403) {
          if (errorData.error?.errors?.[0]?.reason === 'quotaExceeded') {
            throw new YouTubeApiError(
              'YouTube API quota exceeded. Please try again later.',
              403,
              'quotaExceeded',
              errorData
            );
          } else if (errorData.error?.errors?.[0]?.reason === 'forbidden') {
            throw new YouTubeApiError(
              'Access forbidden. Check your API key permissions.',
              403,
              'forbidden',
              errorData
            );
          }
        } else if (response.status === 404) {
          throw new YouTubeApiError(
            'Video not found or not a live stream.',
            404,
            'notFound',
            errorData
          );
        } else if (response.status === 400) {
          throw new YouTubeApiError(
            'Invalid request. Check video ID and API key.',
            400,
            'badRequest',
            errorData
          );
        } else if (response.status === 429) {
          // Rate limit exceeded - implement exponential backoff
          if (retryCount < this.maxRetries) {
            const delay = this.rateLimitDelay * Math.pow(this.backoffMultiplier, retryCount);
            console.warn(`Rate limit exceeded, retrying in ${delay}ms...`);
            await new Promise(resolve => setTimeout(resolve, delay));
            return this.makeRequest<T>(url, retryCount + 1);
          } else {
            throw new YouTubeApiError(
              'Rate limit exceeded. Please try again later.',
              429,
              'rateLimitExceeded',
              errorData
            );
          }
        } else if (response.status >= 500) {
          // Server error - retry with exponential backoff
          if (retryCount < this.maxRetries) {
            const delay = this.rateLimitDelay * Math.pow(this.backoffMultiplier, retryCount);
            console.warn(`Server error, retrying in ${delay}ms...`);
            await new Promise(resolve => setTimeout(resolve, delay));
            return this.makeRequest<T>(url, retryCount + 1);
          } else {
            throw new YouTubeApiError(
              'YouTube API server error. Please try again later.',
              response.status,
              'serverError',
              errorData
            );
          }
        }

        throw new YouTubeApiError(
          `YouTube API error: ${response.status} ${response.statusText}`,
          response.status,
          'apiError',
          errorData
        );
      }

      return await response.json();
    } catch (error) {
      if (error instanceof YouTubeApiError) {
        throw error;
      }
      
      // Network or other errors
      if (retryCount < this.maxRetries) {
        const delay = this.rateLimitDelay * Math.pow(this.backoffMultiplier, retryCount);
        console.warn(`Network error, retrying in ${delay}ms...`, error);
        await new Promise(resolve => setTimeout(resolve, delay));
        return this.makeRequest<T>(url, retryCount + 1);
      }
      
      throw new YouTubeApiError(
        `Network error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        0,
        'networkError',
        error
      );
    }
  }

  async validateApiKey(): Promise<boolean> {
    try {
      const url = `${this.baseUrl}/search?part=snippet&maxResults=1&q=test&key=${this.apiKey}`;
      await this.makeRequest(url);
      return true;
    } catch (error) {
      console.error('API key validation failed:', error);
      return false;
    }
  }

  async getVideoDetails(videoId: string): Promise<VideoResponse> {
    const url = `${this.baseUrl}/videos?part=liveStreamingDetails&id=${videoId}&key=${this.apiKey}`;
    return this.makeRequest<VideoResponse>(url);
  }

  async getLiveChatId(videoId: string): Promise<string> {
    const videoResponse = await this.getVideoDetails(videoId);
    
    if (!videoResponse.items || videoResponse.items.length === 0) {
      throw new YouTubeApiError(
        'Video not found. Please check the video ID.',
        404,
        'videoNotFound'
      );
    }

    const video = videoResponse.items[0];
    if (!video.liveStreamingDetails) {
      throw new YouTubeApiError(
        'This video is not a live stream.',
        400,
        'notLiveStream'
      );
    }

    const liveChatId = video.liveStreamingDetails.activeLiveChatId;
    if (!liveChatId) {
      throw new YouTubeApiError(
        'Live chat is not available for this stream. It may be disabled or the stream may have ended.',
        400,
        'liveChatNotAvailable'
      );
    }

    return liveChatId;
  }

  async getLiveChatMessages(
    liveChatId: string, 
    pageToken?: string
  ): Promise<LiveChatResponse> {
    let url = `${this.baseUrl}/liveChat/messages?liveChatId=${liveChatId}&part=snippet,authorDetails&key=${this.apiKey}`;
    
    if (pageToken) {
      url += `&pageToken=${pageToken}`;
    }

    return this.makeRequest<LiveChatResponse>(url);
  }

  async *getSuperChats(videoId: string): AsyncGenerator<LiveChatMessage[], void, unknown> {
    try {
      // Get the live chat ID
      const liveChatId = await this.getLiveChatId(videoId);
      console.log('Connected to live chat:', liveChatId);

      let pageToken: string | undefined;
      let consecutiveErrors = 0;
      const maxConsecutiveErrors = 5;

      while (true) {
        try {
          const response = await this.getLiveChatMessages(liveChatId, pageToken);
          consecutiveErrors = 0; // Reset error counter on success

          // Filter for super chats and fan funding events
          const superChats = response.items.filter(item => 
            item.snippet.type === 'superChatEvent' || 
            item.snippet.type === 'fanFundingEvent'
          );

          if (superChats.length > 0) {
            yield superChats;
          }

          // Update page token for next request
          pageToken = response.nextPageToken;

          // Wait for the polling interval specified by YouTube
          const pollingInterval = Math.max(response.pollingIntervalMillis || 5000, 1000);
          await new Promise(resolve => setTimeout(resolve, pollingInterval));

        } catch (error) {
          consecutiveErrors++;
          console.error(`Error fetching live chat messages (attempt ${consecutiveErrors}):`, error);

          if (error instanceof YouTubeApiError) {
            // Handle specific errors
            if (error.code === 'quotaExceeded') {
              throw error; // Don't retry quota exceeded errors
            } else if (error.code === 'forbidden' || error.code === 'badRequest') {
              throw error; // Don't retry permission/validation errors
            } else if (error.code === 'notFound') {
              throw new YouTubeApiError(
                'Live stream ended or chat is no longer available.',
                404,
                'streamEnded'
              );
            }
          }

          if (consecutiveErrors >= maxConsecutiveErrors) {
            throw new YouTubeApiError(
              `Too many consecutive errors (${consecutiveErrors}). Stopping connection.`,
              0,
              'tooManyErrors',
              error
            );
          }

          // Exponential backoff for retries
          const backoffDelay = Math.min(
            this.rateLimitDelay * Math.pow(this.backoffMultiplier, consecutiveErrors - 1),
            30000 // Max 30 seconds
          );
          console.warn(`Retrying in ${backoffDelay}ms...`);
          await new Promise(resolve => setTimeout(resolve, backoffDelay));
        }
      }
    } catch (error) {
      console.error('Fatal error in getSuperChats:', error);
      throw error;
    }
  }

  // Utility method to extract message text from super chat
  static extractMessageText(message: LiveChatMessage): string {
    if (message.snippet.superChatDetails?.userComment) {
      return message.snippet.superChatDetails.userComment;
    } else if (message.snippet.fanFundingEventDetails?.userComment) {
      return message.snippet.fanFundingEventDetails.userComment;
    } else if (message.snippet.displayMessage) {
      return message.snippet.displayMessage;
    } else if (message.snippet.textMessageDetails?.messageText) {
      return message.snippet.textMessageDetails.messageText;
    }
    return '';
  }

  // Utility method to get super chat amount
  static getSuperChatAmount(message: LiveChatMessage): string {
    if (message.snippet.superChatDetails?.amountDisplayString) {
      return message.snippet.superChatDetails.amountDisplayString;
    } else if (message.snippet.fanFundingEventDetails?.amountDisplayString) {
      return message.snippet.fanFundingEventDetails.amountDisplayString;
    }
    return '';
  }
}
