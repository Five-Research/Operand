import json
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

class GroqWebScraper:
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.max_html_length = 6000  # Groq's Llama 3 70B has 8k context
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
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

    def _create_groq_payload(self, messages: list) -> dict:
        """Construct Groq API payload for Llama 3 70B"""
        return {
            "model": "llama3-70b-8192",
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 1024,
            "top_p": 0.9,
            "stop": ["<|eot_id|>"],
            "stream": False
        }

    def _parse_llm_response(self, response: str) -> dict:
        """Extract JSON actions from LLM response"""
        try:
            json_str = response.split("```json")[1].split("```")[0].strip()
            return json.loads(json_str)
        except (IndexError, json.JSONDecodeError):
            return {"action": "response", "content": response}

    def execute_agent_loop(self, user_query: str) -> str:
        """Main agent execution flow"""
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
                # Get LLM response from Groq
                response = self.session.post(
                    self.base_url,
                    json=self._create_groq_payload(conversation_history)
                ).json()
                
                llm_message = response["choices"][0]["message"]["content"]
                print(f"\n[LLM STEP]\n{llm_message}\n")
                
                # Parse and execute action
                action = self._parse_llm_response(llm_message)
                
                if action["action"] == "response":
                    return action["content"]
                
                elif action["action"] == "fetch":
                    result = self.fetch_website(action["url"])
                    conversation_history.append({
                        "role": "assistant",
                        "content": f"Fetched content from {action['url']}:\n{result}"
                    })
                
                elif action["action"] == "render":
                    result = self.fetch_dynamic_content(
                        action["url"],
                        action.get("wait_for")
                    )
                    conversation_history.append({
                        "role": "assistant",
                        "content": f"Rendered content from {action['url']}:\n{result}"
                    })
                
                elif action["action"] == "extract":
                    result = self.extract_data(
                        action["html"],
                        action["selector"]
                    )
                    conversation_history.append({
                        "role": "assistant",
                        "content": f"Extraction results using '{action['selector']}':\n{json.dumps(result)}"
                    })
                
                else:
                    return "ERROR: Invalid action requested"
            
            except Exception as e:
                return f"AGENT_ERROR: {str(e)}"

        return "MAX_STEPS_REACHED: Processing limit exceeded"

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
{"action": "extract", "html": "<content>", "selector": "<css_selector>"}
```
- Final response: 
```json
{"action": "response", "content": "<final_answer>"}
```

**Workflow Rules:**
1. Always start with basic fetch before rendering
2. Verify content contains target data elements
3. Use precise CSS selectors (classes > tags)
4. Handle pagination if needed
5. Validate extracted data format
6. Retry with alternatives on failure

**Response Protocol:**
- Use only JSON-formatted code blocks for actions
- Include detailed error diagnostics
- Maintain chain of thought reasoning
- Validate all extracted data
"""

if __name__ == "__main__":
    # Initialize with your Groq API key
    scraper = GroqWebScraper(groq_api_key="YOUR-API-KEY-HERE")
    
    # Example query
    result = scraper.execute_agent_loop(
        "what are the top trending posts on reddit about biohacking"
    )
    
    print("\nFinal Result:")
    print(result)
