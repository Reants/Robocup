from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer


bob = ChatBot("Bob",logic_adapters=["chatterbot.logic.BestMatch","chatterbot.logic.TimeLogicAdapter"])


while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "bye"]:
        
        print("Have a nice day. Goodbye!")
        break
    
    response = bob.get_response(user_input)
    print(f"Bob:",str(response))