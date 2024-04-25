import discord
import os
import torch
from discord.ext import commands
from transformers import AutoModelForCausalLM, AutoTokenizer

intents = discord.Intents.default()
intents.message_content = True

# Grab token from text file
token_file_path = 'bot_token.txt'
token = ''
with open(token_file_path, 'r') as file:
    token = file.read().strip()

# Initialize your bot. Bot can be summoned with "!name"
bot = commands.Bot(command_prefix='!', intents=intents)

# Load your pre-trained model and tokenizer
model_name = 'gpt2'  # Replace with the desired model (e.g., 'gpt2', 'gpt2-medium', 'gpt2-large', etc.)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name} ({bot.user.id})')

@bot.command(name='noodlemaster')
async def generate_response(ctx, *, prompt: str):
    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate a response from the model
    with torch.no_grad():
        output = model.generate(input_ids, max_length=50, num_return_sequences=1)

    # Decode the generated response
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Send the response back to the channel
    await ctx.send(response)
    # Run your bot
    bot.run(token)

"""Install the required libraries (discord.py and transformers) using
pip install discord.py transformers.

Customize the model_name variable to the specific Hugging Face model you want to use (e.g., ‘gpt2’, ‘gpt2-medium’, etc.).

When a user types !noodlemaster "prompt goes here" in a Discord channel, the bot will generate a response based on the provided prompt using the specified model.
"""

