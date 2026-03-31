from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama.chat_models import ChatOllama
from langchain_community.chat_message_histories import FileChatMessageHistory

llm = llm = ChatOllama(model="llama3.2:latest")

prompt_temp = PromptTemplate(
    template="""
            You are an expert in marketing and brand strategy. 
            My business name is {business_name} 
            and my business description is {business_description}.
            Create a compelling brand slogan for a tech startup that specializes in AI-powered solutions for small businesses. 
            The slogan should be memorable, concise, and reflect the company's mission to democratize AI technology. 
            Provide 3 slogan options and explain the reasoning behind each choice.             
            """,
    input_variables=["business_name", "business_description"],
    template_format="f-string",
    # partial_variables={
    #     "business_description": "AI-powered solutions for small businesses"
    # },
)
prompt = prompt_temp.format(business_name="ZenAI" 
, business_description="AI-powered solutions for small businesses")
result = llm.invoke(prompt)
# print(result.content)


# ---------------------------------------------------------------------------------------------

file_path = './langchain/prompts/chat_history.json'
history = FileChatMessageHistory(file_path)
history.add_message(SystemMessage(content='You are helpful assistance, which help in planning vacations'))

while True:
    query = input("You: ")
    if query == "exit":
        break
    else:
        history.add_message(HumanMessage(content=query))
        result = llm.invoke(history.messages)
        history.add_message(AIMessage(content=result.content))
        print("AI: ", result.content)

# print('messages', messages)

# ---------------------------------------------------------------------------------------------

# prompt_temp.save('template.json')

loaded_prompt = PromptTemplate.from_file('./langchain/prompts/template.json')
result = llm.invoke(prompt)
# print(result.content)

# ---------------------------------------------------------------------------------------------

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. Your name is {name}"),
    ("human", "{user_input}"),
])

chain = chat_prompt | llm
result = chain.invoke({"name": "Bob", "user_input": "What is your name?"})
print(result.content)
