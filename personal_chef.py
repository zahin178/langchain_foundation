from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage
from langchain.tools import tool
from ddgs import DDGS

load_dotenv()

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo to get real-time information."""

    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=max_results)]
        if not results:
            return "No results found."
        
        search_results = []

        for i,result in enumerate(results):
            formatted_item = (
                f"Result #{i+1}\n"
                f"Title: {result.get('title')}\n"
                f"URL: {result.get('href')}\n"
                f"Snippet: {result.get('body')}\n"
            )
            search_results.append(formatted_item)
        
        return "\n---\n".join(search_results)

if __name__ == '__main__':
    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

    with open('prompts/bengali_chef_prompt.md', 'r') as file:
        prompt = file.read()

    system_prompt = prompt
    chef_agent = create_agent(model=model, tools=[web_search], system_prompt=system_prompt)

    messages = {'messages':[
        HumanMessage(content="Introduce yourself")
    ]}
    response = chef_agent.invoke(messages)

    print(f"AI: {response['messages'][1].content}\n")
    
    while True:
        user_input = input('User: ')
        if user_input.lower() == 'exit':
            break
        messages['messages'].append(HumanMessage(content=user_input))
        response = chef_agent.invoke(messages)
        print(f"AI: {response['messages'][-1].content}\n")
        messages['messages'].append(AIMessage(content=response['messages'][-1].content))