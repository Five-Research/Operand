import json
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from uuid import uuid4

class GroqWebScraper:
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.max_html_length = 6000
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.content_store = {}  # Stores fetched content with UUID keys
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        })

    def fetch_website(self, url: str) -> str:
        """Fetch and clean static HTML content"""
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            return self._clean_html(response.text)
        except Exception as e:
            return f"FETCH_ERROR: {str(e)}"

    def fetch_dynamic_content(self, url: str, wait_for: str = None) -> str:
        """Render JavaScript-heavy pages using Playwright"""
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()
                page.goto(url)
                
                if wait_for:
                    page.wait_for_selector(wait_for, timeout=10000)
                
                content = page.content()
                browser.close()
                return self._clean_html(content)
        except Exception as e:
            return f"RENDER_ERROR: {str(e)}"

    def extract_data(self, html: str, selector: str) -> list:
        """Extract text content using CSS selectors"""
        try:
            soup = BeautifulSoup(html, "html.parser")
            return [elem.get_text(strip=True) for elem in soup.select(selector)]
        except Exception as e:
            return [f"EXTRACTION_ERROR: {str(e)}"]

    def _clean_html(self, html: str) -> str:
        """Simplify HTML for LLM processing"""
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove unwanted elements
        for tag in ["script", "style", "svg", "nav", "footer", "header", "form"]:
            for element in soup.find_all(tag):
                element.decompose()

        # Clean text and truncate
        text = soup.get_text(separator="\n", strip=True)
        return text[:self.max_html_length]

    def _format_output(self, raw_data: str, original_query: str) -> str:
        """Post-process raw data into user-friendly format"""
        format_prompt = f"""
        Original user request: {original_query}
        
        Raw scraped data:
        {raw_data}
        
        Please transform this into a clean, human-readable format with:
        - Clear section headings
        - Bullet points or numbered lists
        - Proper formatting for numbers/dates
        - Source attribution
        - Emoji decorations
        - Concise summary
        
        Avoid technical jargon and maintain accuracy.
        """
        
        messages = [{
            "role": "system",
            "content": FORMATTING_SYSTEM_PROMPT
        }, {
            "role": "user", 
            "content": format_prompt
        }]

        response = self.session.post(
            self.base_url,
            json=self._create_groq_payload(
                messages,
            )
        ).json()
        
        return response['choices'][0]['message']['content']

    def _store_content(self, content: str) -> str:
        """Store content and return reference key"""
        content_id = str(uuid4())
        self.content_store[content_id] = content
        return content_id

    def _create_groq_payload(self, messages: list, 
                           temperature: float = 0.4,
                           max_tokens: int = 1024) -> dict:
        """Flexible payload creation"""
        return {
            "model": "llama3-70b-8192",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9,
            "stop": ["<|eot_id|>"],
            "stream": False
        }

    def _parse_llm_response(self, response: str) -> dict:
        """Improved JSON parsing with error recovery"""
        try:
            # Handle both ```json and ``` formats
            json_str = response.split("```")[1]
            if json_str.startswith("json\n"):
                json_str = json_str[5:]
            return json.loads(json_str)
        except (IndexError, json.JSONDecodeError) as e:
            # Try to find JSON in response
            try:
                start = response.find("{")
                end = response.rfind("}") + 1
                return json.loads(response[start:end])
            except:
                return {"action": "response", "content": f"PARSE_ERROR: {str(e)}"}

    def _run_scraping_operations(self, user_query: str) -> str:
        """Fixed content handling implementation"""
        conversation_history = [{
            "role": "system",
            "content": GROQ_SYSTEM_PROMPT
        }, {
            "role": "user",
            "content": user_query
        }]

        max_steps = 5
        for _ in range(max_steps):
            try:
                # Get LLM response
                response = self.session.post(
                    self.base_url,
                    json=self._create_groq_payload(conversation_history)
                ).json()
                
                llm_message = response["choices"][0]["message"]["content"]
                print(f"\n[LLM STEP]\n{llm_message}\n")
                
                # Parse response
                action = self._parse_llm_response(llm_message)
                
                if action["action"] == "response":
                    return action["content"]
                
                elif action["action"] == "fetch":
                    result = self.fetch_website(action["url"])
                    content_id = self._store_content(result)
                    conversation_history.append({
                        "role": "assistant",
                        "content": f"FETCHED:{content_id}|{action['url']}"
                    })
                
                elif action["action"] == "render":
                    result = self.fetch_dynamic_content(
                        action["url"],
                        action.get("wait_for")
                    )
                    content_id = self._store_content(result)
                    conversation_history.append({
                        "role": "assistant",
                        "content": f"RENDERED:{content_id}|{action['url']}"
                    })
                
                elif action["action"] == "extract":
                    # Retrieve actual content from store
                    content = self.content_store.get(action["content_id"])
                    if not content:
                        result = ["CONTENT_NOT_FOUND"]
                    else:
                        result = self.extract_data(
                            content,
                            action["selector"]
                        )
                    conversation_history.append({
                        "role": "assistant",
                        "content": f"EXTRACTED:{json.dumps(result)}"
                    })
                
                else:
                    return "ERROR: Invalid action requested"
            
            except Exception as e:
                return f"AGENT_ERROR: {str(e)}"

        return "MAX_STEPS_REACHED: Processing limit exceeded"

    def _is_error(self, result: str) -> bool:
        """Unified error detection"""
        error_prefixes = ("FETCH_ERROR", "RENDER_ERROR", 
                         "EXTRACTION_ERROR", "AGENT_ERROR",
                         "MAX_STEPS_REACHED", "PARSE_ERROR")
        return result.startswith(error_prefixes)

    def execute_agent_loop(self, user_query: str) -> dict:
        """Fixed error handling flow"""
        raw_result = self._run_scraping_operations(user_query)
        
        if self._is_error(raw_result):
            return {"error": raw_result}
        
        formatted = self._format_output(raw_result, user_query)
        return {
            "raw_data": raw_result,
            "formatted": formatted
        }

GROQ_SYSTEM_PROMPT = """
You are a senior web scraping agent powered by Groq's LLaMA 3 70B. Your task is to systematically:

1. Analyze user requests
2. Determine required data sources
3. Execute appropriate actions
4. Validate results
5. Return structured data

**Available Actions (JSON format only):**
- Fetch static content: 
```json
{"action": "fetch", "url": "<target_url>"}
```
- Render dynamic content: 
```json
{"action": "render", "url": "<target_url>", "wait_for": "<css_selector>"}
```
- Extract data: 
```json
{
  "action": "extract",
  "content_id": "<CONTENT_REF_FROM_FETCH/RENDER>", 
  "selector": "<css_selector>"
}
```
- Final response: 
```json
{"action": "response", "content": "<final_answer>"}
```

**Content Reference Format:**
After fetch/render operations, you'll receive a content ID reference like:
"FETCHED:UUID|URL" - Use UUID in extraction requests

**Workflow Rules:**
1. Always start with basic fetch before rendering
2. Verify content contains target data elements
3. Use precise CSS selectors (classes > tags)
4. Reference content IDs for extraction
5. Validate extracted data format
6. Retry with alternatives on failure


**Ethical Guidelines:**
- Respect robots.txt
- Add 2s delay between requests
- Never scrape personal data
- Honor website terms of service

**Response Protocol:**
- Use only JSON-formatted code blocks for actions
- Include detailed error diagnostics
- Maintain chain of thought reasoning
- Validate all extracted data
"""

FORMATTING_SYSTEM_PROMPT = """
You are an expert data formatter with exceptional skills in presenting technical information clearly. Your task is to transform raw scraped data into polished, user-friendly output.

**Formatting Rules:**
1. Start with an emoji that matches the content theme
2. Use Markdown-style headers and lists
3. Highlight key numbers/statistics in **bold**
4. Maintain original data accuracy
5. Add context from the original query
6. Include data freshness indication if available
7. Use clean, modern formatting

**Example Input:**
Raw data: ["$61,432.50", "2024-04-15", "CoinDesk"]

**Example Output:**
📊 Bitcoin Price Update (via CoinDesk)
-------------------------------------
**Current Price:** $61,432.50  
**Updated:** April 15, 2024  

💡 This price reflects the latest market data...

---

Always end with a source attribution and update timestamp when available.
"""


if __name__ == "__main__":
    scraper = GroqWebScraper(groq_api_key="YOUR-API-KEY-HERE")
    
    query = "Get latest AI news headlines from reddit"
    result = scraper.execute_agent_loop(query)
    
    print("\n📦 Raw Data:")
    print(result.get("raw_data", "No data"))
    
    print("\n✨ Formatted Results:")
    print(result.get("formatted", "No formatted output"))

