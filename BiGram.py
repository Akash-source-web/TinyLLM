#importing Dependencies
import torch 
import torch.nn as nn
from torch.nn import functional as f

#Hyperperameters 
batch_size = 32 # how many independent sequences will we process in parallel?
chunk_size = 8  # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
lr = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32

# Modeling --------------------------------
 
torch.manual_seed(1337)

# Load the input data
with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

#unique char available in corpus
chars = sorted(list(set(text)))
vocab_size = len(chars)

#create character level tokenizer
sentToInt = {ch:i for i , ch in enumerate(chars)}
intToSent = {i:ch for i , ch in enumerate(chars)}
encoder = lambda s: [sentToInt[c] for c in s] # take a string, output a list of integer
decoder = lambda l: ''.join([intToSent[i] for i in l]) # takes a list of integer, output a string

# toknize whole training dataset and store it in a tensor
data = torch.tensor(encoder(text), dtype=torch.long)

#split data into train and validation split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

#model recive data in batch which contain multiple chunks of that data for eficenty work in parallel format as transformer use to work
def get_batch(split):
  #create small batches of data of input x and traget y
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - chunk_size, (batch_size,))
  x = torch.stack([data[i:i+chunk_size] for i in ix])
  y = torch.stack([data[i+1:i+chunk_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y

#store the loss into the torch so that no remember of all the loss only last loss will remember and update accordingly in back propogation
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

#creating a sinple neural network called bigram to map each word to each other
class BigramLanguageModel(nn.Module):

  def __init__(self):
    #each token directly read off the logits for the next token from a loockup table
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embed) #n_embed- Number of embeding dimentions

  def forward(self, idx, target=None):
    #idx and target are both (B,T) tensor of integers
    logits = self.token_embedding_table(idx)  #(B,T,C)
    if target is None:
      loss = None
    else:
      #Calculate loss of the model for good learning
      #Convert our shape to (B,C) by combining the B and T in  1 dimention
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      target = target.view(B*T)
      loss = f.cross_entropy(logits, target) # cross entropy except the shape in (B,C,T) or (B,C)
    return logits, loss

  def generator(self, idx, max_new_token):
    #idx is (B,T) array of indices in the corrent context
    for _ in range(max_new_token):
      # get the predictions
      logits, loss = self(idx)
      #focus only on the last time step
      logits = logits[:, -1, :] # becomes (B, C)
      # apply softmax function to get probabilities
      probs = f.softmax(logits, dim=1) #(B,C)
      #Sample from the distribuations
      idx_next = torch.multinomial(probs, num_samples=1) #(B, 1)
      #append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) #(B, T+1)
    return idx

model = BigramLanguageModel()
m = model.to(device)

# Creating a pytorch optimizer to update the loss
optimizer = torch.optim.AdamW(m.parameters(), lr=lr)

#creating a function to update parameter to lower the loss in back propogation
for iter in range(max_iters):
  # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # sample a batch of data
    xb, yb = get_batch('train')
    
    #evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decoder(m.generator(context, max_new_token=500)[0].tolist()))