from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import re

def preprocess_text(text):
    # Remove date and time stamps
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2} [AP]M:', '', text)
    # Remove names
    text = re.sub(r'[^\n:]+:', '', text)
    return text


def summarize_text(text, target_percentage=0.1):
    # Parse the text
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    
    # Get the total number of sentences in the original text
    total_sentences = len(parser.document.sentences)
    
    # Calculate the target number of sentences for the summary
    target_sentences = int(total_sentences * target_percentage)
    
    # Create the LSA summarizer
    summarizer = LsaSummarizer()
    
    # Summarize the text with the target number of sentences
    summary_sentences = summarizer(parser.document, target_sentences)
    
    # Combine the summary sentences into a single string
    summary_text = " ".join(str(sentence) for sentence in summary_sentences)
    
    return summary_text

