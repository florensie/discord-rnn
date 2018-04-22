import discord
from discord import ChannelType
from discord.ext import commands
from textgenrnn import textgenrnn
from termcolor import colored
import yaml


with open("config.yml", "r") as f:
    cfg = yaml.load(f)


MSG_LIMIT = 1000000
# For printing out to cli
DIV_S = '-' * 15
DIV_L = '-' * 40


bot = commands.Bot(command_prefix='!',
                   description="Bot that collects all of a servers' messages and tries to replicate them")


@bot.event
async def on_ready():
    global appinfo
    appinfo = await bot.application_info()

    # Show bot info
    print('Logged in as:')
    print(bot.user)
    print(bot.user.id)
    print(DIV_S)
    print('Bot owner:')
    print(appinfo.owner)
    print(appinfo.owner.id)
    print(DIV_L)

    # Retrieve messages
    texts = []
    context_labels = []
    print('Retrieving messages')
    await easy_presence('Retrieving messages')
    n = {'total': 0}
    for server in bot.servers:
        if server.name in cfg['discord_servers']:
            n['server'] = 0
            print_indent(colored(server.name + ': ', 'blue'), 1)
            for channel in server.channels:
                # First check if the channel is a text channel and we are allowed to read messages
                if channel.type == ChannelType.text and channel.permissions_for(server.me).read_messages:
                    print_indent(channel.name + ': ', 2, end='', flush=True)
                    n['channel'] = 0
                    messages = bot.logs_from(channel, MSG_LIMIT)
                    async for message in messages:
                        if message.content != '':
                            texts.append(message.content)
                            context_labels.append(server.name)
                        n['channel'] += 1
                    if n['channel'] == MSG_LIMIT:
                        print_indent(colored('We hit our message limit!', 'red'), 2)
                    print(n['channel'])
                    n['server'] += n['channel']
            print_indent(colored('Total: ' + str(n['server']), 'yellow'), 2)
            n['total'] += n['server']
    print_indent(colored('Total: ' + str(n['total']), 'yellow'), 1)

    # Train neural net
    print('Training RNN')
    await easy_presence('Training RNN')

    textgen = textgenrnn(name='{}_discord'.format("_".join(cfg['discord_servers'])))
    if cfg['new_model']:
        textgen.train_new_model(
            texts,
            context_labels=context_labels,
            num_epochs=cfg['num_epochs'],
            gen_epochs=cfg['gen_epochs'],
            batch_size=cfg['batch_size'],
            prop_keep=cfg['prop_keep'],
            rnn_layers=cfg['model_config']['rnn_layers'],
            rnn_size=cfg['model_config']['rnn_size'],
            rnn_bidirectional=cfg['model_config']['rnn_bidirectional'],
            max_length=cfg['model_config']['max_length'],
            dim_embeddings=cfg['model_config']['dim_embeddings'],
            word_level=cfg['model_config']['word_level'])
    else:
        textgen.train_on_texts(
            texts,
            context_labels=context_labels,
            num_epochs=cfg['num_epochs'],
            gen_epochs=cfg['gen_epochs'],
            prop_keep=cfg['prop_keep'],
            batch_size=cfg['batch_size'])


async def easy_presence(message: str):
    """Makes changing the presence string easier"""
    await bot.change_presence(game=discord.Game(name=message))


def print_indent(message: str, level: int=0, indent: int=4, *args, **kwargs):
    """Prints a message with an indent"""
    indent_str = ' ' * indent * level
    print(indent_str + message, *args, **kwargs)


bot.run(cfg['token'])
