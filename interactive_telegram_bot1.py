#!/usr/bin/env python3
#
# Judas Ear prototype 002

import json
import os
import numpy as np
import tensorflow as tf
import logging
import re
import string
from itertools import groupby
from telegram import (ReplyKeyboardMarkup, ReplyKeyboardRemove)
from telegram.ext import (Updater, CommandHandler, MessageHandler, Filters, RegexHandler,
                          ConversationHandler)
import model, sample, encoder

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

TALK, START, REPLYTALK, REPLYRAW = range(4)

# her name
my_name = 'Kikurage'
# telegram api key
apikey = ''
# telegram user who will be allowed to talk to her
owner='@markhorovitz'
persistent_context=' '
sample_context=''


model_name='774M'
seed=None
nsamples=1
length=256 # 128 is also fine (and faster)
temperature=1
top_k=16 # 16 is ok too
batch_size = 1 # leftovers
assert nsamples % batch_size == 0
output = None
sess = None

def compress(istring):
    return  re.sub('[%s]' % string.digits, '',  ''.join('%s%s' % (char, sum(1 for _ in group)) for char, group in groupby(istring)))

def start(update, context):
    update.message.reply_text(
        'Send /cancel to stop talking to me.\n\n'
        'Send /raw to enter raw queries.\n\n'
        'Send /talk to enter context talk mode. \n\n'
        '')
    return START


def talk(update, context):
    logger.info("TALK mode started")
    update.message.reply_text('TALK MODE START. enter your text or /cancel',
                              reply_markup=ReplyKeyboardRemove())

    return REPLYTALK

def raw(update, context):

    logger.info("RAW mode started")
    update.message.reply_text('RAW MODE START. enter your text or /cancel',
                              reply_markup=ReplyKeyboardRemove())

    return REPLYRAW

def replytalk(update, context):
    # carry global vars
    global length
    global sess
    global output
    global model_name
    global batch_size
    global nsamples
    global tfcontext
    global output
    global sample_context
    global persistent_context
    global my_name

    user = update.message.from_user
    logger.info("REPLYTALK received of %s: %s", user.first_name, update.message.text)

    if not sample_context:
        # initialize. try to inject some context to hint the model
        sample_context = 'Conversation of ' + my_name + ', and a person from internet called '  + user.first_name + '.\n'
        sample_context =  sample_context + persistent_context +'\n\n'
        sample_context = sample_context + my_name + ' - Hi ' + user.first_name + '\n'

    raw_text = update.message.text
    sample_context = sample_context + user.first_name + ' - ' + raw_text + '\n' 
   
    enc = encoder.get_encoder(model_name)
    context_tokens = encoder.get_encoder(model_name).encode(sample_context)
    logger.info("sample_context: " + sample_context )
    logger.info("sample_context_len: " + str(len(context_tokens)))
    out = sess.run(output, feed_dict={ tfcontext: [context_tokens for _ in range(1)] })[:, len(context_tokens):]
    text = enc.decode(out[0])
    logger.info("Model run complete")

    # parse the response somehow
    logger.info("model response" + text)
    logger.info("first line response" + text.split('\n')[0])
    model_response_text = ''

    if len(text.split('\n')[0]) < 5 or len(compress(text.split('\n')[0])) < 5:
        model_response_text =  text.split('\n')[1].lstrip() #+ '\n'
    else:
        model_response_text =  text.split('\n')[0].lstrip() #+ '\n' 

    logger.info("guessed response" + model_response_text)

    # if model response starts with correspondent name...
    if (model_response_text.startswith(user.first_name)):
    # v002+ just look for the first line beginning with my name
        for line in text.split('\n'):
            if line.startswith(my_name + ' - '):
                model_response_text = line.split('-')[1]
                logger.info("guessed response (2)" + model_response_text)

    if '<|endoftext|>' in model_response_text:
        model_response_text = model_response_text.split('<|endoftext|>')[0]

    # sometimes my name is mentioned on line 1 need to clean that
    if model_response_text.startswith(my_name + ' - '):
        model_response_text = model_response_text.split(my_name + ' - ')[1]

    logger.info("final response " + model_response_text)

    update.message.reply_text(model_response_text,
                      reply_markup=ReplyKeyboardRemove())

    sample_context = sample_context + my_name + ' - ' + model_response_text + '\n'

    # truncate the context
    linecount = 0
    count = 0
    for line in sample_context.splitlines():
        linecount += 1
    logger.info("ctx length " + str(linecount) + " " + str(len(context_tokens)) + " tokens")
    if linecount > 30 or len(context_tokens) > 800:
        #sample_context_new = '';
        sample_context_new = persistent_context + '\n\n'
        for line in sample_context.splitlines():
            count += 1
            if count > (linecount - 30):
                sample_context_new = sample_context_new + line + '\n'
        sample_context = sample_context_new

    return REPLYTALK

def replyraw(update, context):
    # carry global vars
    global length
    global sess
    global output
    global model_name
    global batch_size
    global nsamples
    global tfcontext
    global output



    user = update.message.from_user
    logger.info("RAW query received of %s: %s", user.first_name, update.message.text)
    raw_text = update.message.text
    enc = encoder.get_encoder(model_name)
    context_tokens = encoder.get_encoder(model_name).encode(raw_text)
    generated = 0
    for _ in range(nsamples // batch_size):
        out = sess.run(output, feed_dict={
            tfcontext: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]
        for i in range(batch_size):
            generated += 1
            text = enc.decode(out[i])
            logger.info("Model run complete")
            logger.info("model response" + text)
            update.message.reply_text(text,
                              reply_markup=ReplyKeyboardRemove())
    return REPLYRAW


def cancel(update, context):
    global sample_context
    sample_context = ''
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)
    update.message.reply_text('Bye! I hope we can talk again some day.',
                              reply_markup=ReplyKeyboardRemove())
    logger.info("sample_context: " + sample_context)

    return ConversationHandler.END

def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def main():

    # carry global vars
    global apikey
    global length
    global sess
    global output
    global length
    global sess
    global output
    global model_name
    global batch_size
    global nsamples
    global tfcontext
    global owner

    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    updater = Updater(apikey, use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start, Filters.user(username=owner))],

        states={
            TALK: [MessageHandler(Filters.text, talk)],
            REPLYTALK: [MessageHandler(Filters.text, replytalk)],
            REPLYRAW: [MessageHandler(Filters.text, replyraw)]
        },

        fallbacks=[CommandHandler('cancel', cancel),
                   CommandHandler('raw', raw),
                    CommandHandler('talk', talk)]
    )

    dp.add_handler(conv_handler)
    dp.add_error_handler(error)

    
    # initialize the model, etc
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))
    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
    with tf.Session(graph=tf.Graph()) as sess:
        tfcontext = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=tfcontext,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)
        logger.info("Model initialized")
        # Start the Bot
        updater.start_polling()

        # Run the bot until you press Ctrl-C or the process receives SIGINT,
        # SIGTERM or SIGABRT. This should be used most of the time, since
        # start_polling() is non-blocking and will stop the bot gracefully.
        updater.idle()


if __name__ == '__main__':
    main()

