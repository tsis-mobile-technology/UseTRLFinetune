"""
Data collector for fetching content from Reddit.
"""
import os
import logging
import praw
from praw.exceptions import RedditAPIException, APIException
from prawcore.exceptions import Redirect, NotFound, Forbidden, PrawcoreException

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('reddit_collector')

# --- Reddit API Credentials ---
# 중요: 실제 환경에서는 이 값들을 하드코딩하지 마십시오.
# 환경 변수, 설정 파일 또는 안전한 저장소를 사용하세요.
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "YOUR_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "YOUR_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "YOUR_USER_AGENT (e.g., MyBot/0.1 by u/YourUsername)")

class RedditCollector:
    def __init__(self, client_id, client_secret, user_agent):
        """
        Initializes the Reddit API client.

        Args:
            client_id (str): Reddit API client ID.
            client_secret (str): Reddit API client secret.
            user_agent (str): Reddit API user agent.
        """
        if not all([client_id, client_secret, user_agent]) or \
           client_id == "YOUR_CLIENT_ID" or client_secret == "YOUR_CLIENT_SECRET" or user_agent == "YOUR_USER_AGENT":
            logger.warning("Reddit API credentials are not fully configured. Using read-only mode for public subreddits only.")
            # For read-only mode with PRAW, a client_id and user_agent are still needed.
            # client_secret being None (or not provided) triggers read-only mode.

            # Use a default user_agent if the provided one is the placeholder
            effective_user_agent = user_agent
            if user_agent == "YOUR_USER_AGENT":
                effective_user_agent = "PRAW_Read_Only/0.1 by PythonScript"

            # Pass client_id even if it's the placeholder; PRAW needs it.
            # If client_secret is the placeholder, pass None to trigger read-only.
            self.reddit = praw.Reddit(
                client_id=client_id if client_id != "YOUR_CLIENT_ID" else "PLACEHOLDER_ID", # PRAW requires a non-None client_id.
                                                                                          # Using a generic placeholder if actual is missing.
                                                                                          # This might still be rejected by Reddit API if not a valid format.
                client_secret=client_secret if client_secret != "YOUR_CLIENT_SECRET" else None,
                user_agent=effective_user_agent,
            )
        else:
            # All credentials seem to be provided and are not the default placeholders
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
        logger.info("Reddit PRAW client initialized.")
        if self.reddit.read_only:
            logger.info("PRAW client is in read-only mode.")
        else:
            logger.info("PRAW client is in authenticated mode (read-write potential).")


    def get_subreddit_posts(self, subreddit_name, limit=10, sort_by='hot'):
        """
        Fetches posts from a specified subreddit.

        Args:
            subreddit_name (str): The name of the subreddit (e.g., 'python').
            limit (int): Maximum number of posts to fetch.
            sort_by (str): Sorting method ('hot', 'new', 'top', 'controversial').

        Returns:
            list: A list of dictionaries, where each dictionary contains post data.
                  Returns an empty list if an error occurs or subreddit is not found.
        """
        posts_data = []
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            logger.info(f"Fetching {limit} posts from r/{subreddit_name} sorted by '{sort_by}'...")

            if sort_by == 'hot':
                submissions = subreddit.hot(limit=limit)
            elif sort_by == 'new':
                submissions = subreddit.new(limit=limit)
            elif sort_by == 'top':
                submissions = subreddit.top(limit=limit)
            elif sort_by == 'controversial':
                submissions = subreddit.controversial(limit=limit)
            else:
                logger.error(f"Invalid sort_by value: {sort_by}. Defaulting to 'hot'.")
                submissions = subreddit.hot(limit=limit)

            for submission in submissions:
                post = {
                    'id': submission.id,
                    'title': submission.title,
                    'score': submission.score,
                    'url': submission.url,
                    'permalink': f"https://www.reddit.com{submission.permalink}",
                    'created_utc': submission.created_utc,
                    'selftext': submission.selftext,
                    'num_comments': submission.num_comments,
                    'author': submission.author.name if submission.author else "[deleted]"
                }
                posts_data.append(post)
            logger.info(f"Successfully fetched {len(posts_data)} posts from r/{subreddit_name}.")

        except Redirect:
            logger.error(f"Subreddit r/{subreddit_name} not found or invalid (Redirect).")
        except NotFound:
            logger.error(f"Subreddit r/{subreddit_name} not found (NotFound).")
        except Forbidden:
            logger.error(f"Access to r/{subreddit_name} is forbidden (Forbidden). Ensure credentials are valid for private subreddits.")
        except APIException as e:
            logger.error(f"Reddit API Exception while fetching posts from r/{subreddit_name}: {e}")
        except PrawcoreException as e: # Catches other prawcore exceptions like RequestException
            logger.error(f"PRAW Core Exception while fetching posts from r/{subreddit_name}: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while fetching posts from r/{subreddit_name}: {e}")

        return posts_data

    def get_post_comments(self, post_id=None, permalink=None, limit=20, depth=2):
        """
        Fetches comments from a specific post.
        Provide either post_id or permalink.

        Args:
            post_id (str, optional): The ID of the post.
            permalink (str, optional): The permalink of the post.
            limit (int): Maximum number of top-level comments to fetch.
                         Set to None for PRAW's default (usually more, but can be slow).
            depth (int): How deep to fetch replies (1 for top-level only, 2 for one level of replies, etc.).
                         Higher values significantly increase API calls.

        Returns:
            list: A list of dictionaries, where each dictionary contains comment data.
                  Returns an empty list if an error occurs.
        """
        if not post_id and not permalink:
            logger.error("Either post_id or permalink must be provided to fetch comments.")
            return []

        comments_data = []
        try:
            if post_id:
                submission = self.reddit.submission(id=post_id)
            else: # permalink must be provided
                submission = self.reddit.submission(url=f"https://www.reddit.com{permalink}" if not permalink.startswith("https") else permalink)

            logger.info(f"Fetching comments for post '{submission.title}' (ID: {submission.id}). Limit: {limit}, Depth: {depth}")

            submission.comment_sort = 'top' # or 'new', 'best', etc.
            submission.comments.replace_more(limit=0) # Only fetch top-level comments initially, then expand based on depth

            comment_queue = list(submission.comments[:limit if limit is not None else None])

            current_depth = 1
            processed_comments = 0

            # Basic depth handling - can be improved for more complex scenarios
            # This approach fetches comments level by level up to the specified depth.
            # PRAW's replace_more(limit=None) would fetch all, but can be very slow.
            # Using limit=0 initially and then selectively expanding is more controlled.

            temp_comments = []
            for comment in comment_queue:
                if isinstance(comment, praw.models.MoreComments):
                    # Skipping 'MoreComments' for now in this simplified depth handling
                    # To fetch them: temp_comments.extend(comment.comments())
                    continue

                comment_data = {
                    'id': comment.id,
                    'body': comment.body,
                    'author': comment.author.name if comment.author else "[deleted]",
                    'score': comment.score,
                    'created_utc': comment.created_utc,
                    'permalink': f"https://www.reddit.com{comment.permalink}",
                    'depth': current_depth,
                    'replies': []
                }

                # Fetch replies if depth allows
                if current_depth < depth and hasattr(comment, 'replies'):
                    comment.replies.replace_more(limit=0) # Get direct replies
                    for reply in comment.replies:
                        if isinstance(reply, praw.models.MoreComments):
                            continue
                        reply_data = {
                            'id': reply.id,
                            'body': reply.body,
                            'author': reply.author.name if reply.author else "[deleted]",
                            'score': reply.score,
                            'created_utc': reply.created_utc,
                            'permalink': f"https://www.reddit.com{reply.permalink}",
                            'depth': current_depth + 1
                        }
                        comment_data['replies'].append(reply_data)

                comments_data.append(comment_data)
                processed_comments += 1

            logger.info(f"Successfully fetched {len(comments_data)} top-level comments (with replies up to depth {depth}).")

        except NotFound:
            logger.error(f"Post not found (ID/Permalink: {post_id or permalink}).")
        except Forbidden:
            logger.error(f"Access to post comments is forbidden (ID/Permalink: {post_id or permalink}).")
        except APIException as e:
            logger.error(f"Reddit API Exception while fetching comments: {e}")
        except PrawcoreException as e:
            logger.error(f"PRAW Core Exception while fetching comments: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while fetching comments: {e}")

        return comments_data

if __name__ == "__main__":
    logger.info("Reddit Collector script started for testing.")

    # --- IMPORTANT ---
    # For this example to work, you need to set your Reddit API credentials.
    # 1. Go to https://www.reddit.com/prefs/apps
    # 2. Create a new "script" app.
    # 3. Set a redirect URI (e.g., http://localhost:8080 - it won't be used for script apps but is often required).
    # 4. Get your client ID (under app name) and client secret.
    # 5. User agent can be something like "MyTestBot/0.1 by u/YourRedditUsername"

    # Replace with your actual credentials or set them as environment variables
    # For testing in an environment without real credentials, it will run in read-only mode.

    # These will be picked up from the top of the file, which uses os.getenv
    # For testing, if environment variables are not set, they will default to "YOUR_CLIENT_ID" etc.

    # The __init__ method now handles the placeholder values appropriately.
    collector = RedditCollector(client_id=REDDIT_CLIENT_ID,
                                client_secret=REDDIT_CLIENT_SECRET,
                                user_agent=REDDIT_USER_AGENT)

    if collector.reddit.read_only:
         logger.warning("Collector is in read-only mode due to missing or placeholder full credentials.")
         logger.warning("Functionality may be limited to public subreddits. Ensure User-Agent is descriptive.")
         if REDDIT_CLIENT_ID == "YOUR_CLIENT_ID" or REDDIT_CLIENT_ID == "PLACEHOLDER_ID":
             logger.error("CRITICAL: A valid Client ID is REQUIRED for any Reddit API interaction, even read-only.")
             logger.error("The script will likely fail or be heavily restricted by Reddit.")
             logger.error("Please set REDDIT_CLIENT_ID environment variable or update the script.")


    # Test 1: Fetch posts from a public subreddit
    subreddit_to_fetch = 'python' # A popular and safe subreddit for testing
    logger.info(f"\n--- Test 1: Fetching posts from r/{subreddit_to_fetch} ---")
    posts = collector.get_subreddit_posts(subreddit_to_fetch, limit=3, sort_by='hot')

    if posts:
        for i, post in enumerate(posts):
            print(f"\nPost {i+1}:")
            print(f"  Title: {post['title']}")
            print(f"  Author: {post['author']}")
            print(f"  Score: {post['score']}")
            print(f"  URL: {post['url']}")
            print(f"  Permalink: {post['permalink']}")
            print(f"  Selftext (first 100 chars): {post['selftext'][:100].replace('\n', ' ')}...")

            # Test 2: Fetch comments for the first post
            if i == 0: # Only fetch comments for the first post to keep example concise
                logger.info(f"\n--- Test 2: Fetching comments for post ID {post['id']} ---")
                comments = collector.get_post_comments(post_id=post['id'], limit=2, depth=2)
                if comments:
                    for j, comment in enumerate(comments):
                        print(f"\n  Comment {j+1}:")
                        print(f"    Author: {comment['author']}")
                        print(f"    Score: {comment['score']}")
                        print(f"    Body (first 100 chars): {comment['body'][:100].replace('\n', ' ')}...")
                        if comment['replies']:
                            for k, reply in enumerate(comment['replies']):
                                print(f"      Reply {k+1}:")
                                print(f"        Author: {reply['author']}")
                                print(f"        Body (first 50 chars): {reply['body'][:50].replace('\n', ' ')}...")
                else:
                    print("  No comments fetched or an error occurred.")
    else:
        print(f"No posts fetched from r/{subreddit_to_fetch} or an error occurred.")

    # Test 3: Attempt to fetch from a non-existent subreddit
    logger.info("\n--- Test 3: Fetching posts from a non-existent subreddit ---")
    non_existent_posts = collector.get_subreddit_posts("thisSubredditSurelyDoesNotExist12345", limit=2)
    if not non_existent_posts:
        print("Correctly handled non-existent subreddit (no posts fetched).\n")

    logger.info("Reddit Collector script testing finished.")

# To make this runnable, one would need to:
# 1. `pip install praw`
# 2. Set up Reddit API credentials (client_id, client_secret, user_agent)
#    either as environment variables (REDDIT_CLIENT_ID, etc.) or directly in the script (NOT recommended for production).
#
# Example of running with environment variables:
# export REDDIT_CLIENT_ID="your_id"
# export REDDIT_CLIENT_SECRET="your_secret"
# export REDDIT_USER_AGENT="MyAgent/0.1 by u/yourusername"
# python data_collectors/reddit_collector.py
#
# Without credentials, it will default to read-only mode which PRAW supports for public subreddits.
# Some functionalities might be limited or behave differently in read-only mode.
# For example, accessing very specific user data or private subreddits would fail.
# PRAW automatically handles rate limits by sleeping when necessary.
# Pagination is handled by PRAW's iterators (e.g., subreddit.hot()).
# The `replace_more()` method is used for comments to expand comment trees.
# `limit=0` means "don't fetch any MoreComments objects",
# `limit=None` (default) means "fetch all of them".
# Fetching all can be very slow and API intensive for large comment sections.
# The current comment fetching is a simplified approach.
