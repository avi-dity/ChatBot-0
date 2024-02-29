from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist as scipy_cdist
torch_device='cuda' if torch.cuda.is_available() else 'cpu'
df=pd.read_csv('/Users/abhishekjanjal/test_idea/projectR/rcom/blog/data/game_req.csv')
encoder = SentenceTransformer( 'paraphrase-MiniLM-L6-v2',device=torch_device)
plot_embeddings = encoder.encode(df.name.tolist(),device=torch_device)
def chat_gameR(text):
    game=text
    game_emb=encoder.encode([game],device=torch_device)
    similarities = 1 - scipy_cdist(game_emb,plot_embeddings,'cosine' )
    best_sim_idx = np.argmax(similarities[0])
    most_similar_title = df.loc[best_sim_idx,'name']
    most_similar_plot = df.loc[best_sim_idx].Requirement
    most_similar_title_sim = similarities[0]. max()

    return most_similar_plot



def index(request):
    return render(request,'blog/index.html')


def specific(request):
    return HttpResponse('This is my specific url')

def getResponse(request):
    userMessage=request.GET.get('userMessage')
    chatResponse = chat_gameR(userMessage)
    return HttpResponse(chatResponse)

