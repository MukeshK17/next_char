
import torch
import torch.nn as nn
import streamlit as st
import numpy as np
import time 
import torch._dynamo
torch._dynamo.config.suppress_errors = True

if (torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Define the NextChar model
class NextChar(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.fc1 = nn.Linear(block_size * emb_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 64)
        self.fc3 = nn.Linear(64, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

#Streamlit setup
 
st.write("""
         # Generate the next $k$ characters
         ### Select the Novel""")
dataset = st.selectbox("", ['A Suitable Boy','Malgudi Days', 'Three Men In A Boat', 'Lord of the Flies'])
max_len = st.slider("Select the number of characters to generarte:", min_value=100, max_value=10000, step=100)#, value=1000

# Seed_IN = st.radio("""Do you wish to change the seed number from default 4002?""",
#                    ["Yes", "No"], index=None)
# if Seed_IN == "Yes":
#     seed_n = st.slider("Seed number", 1, 10000)
# else:
#     seed_n = 4002

col1, col2 = st.columns(2)

with col1:
    Seed_IN = st.radio(
        "Do you wish to change the seed number from default 4002?",
        ["Yes", "No"],
        index=None,
        key="seed_choice",
        label_visibility="visible",
        horizontal=True
    )

if Seed_IN == "Yes":
    with col2:
        seed_n = st.slider("Seed number", 1, 10000)
else:
    seed_n = 4002

col1, col2 = st.columns(2)

with col1:
    Embedding_dimension = st.radio(
        "Select the Embedding Dimension:",
        ["20", "40", "80"],
        
        key="Embedding Dimension",
        label_visibility="visible",
        horizontal=True
    )
block_size = 50
if Embedding_dimension == "20":
    emb_dim = 20
elif Embedding_dimension == "40":
    emb_dim = 40
elif Embedding_dimension == "80":
    emb_dim = 80
    block_size = 80

st.sidebar.header(" Next-k Character Generator!")
st.sidebar.caption(
    "This application uses a Multi-Layer Perceptron (MLP) model for next-k character prediction. "
    "It predicts the next possible sequence of characters based on the input provided by the user. "
    "For example, if the input text is *app...*, the model might predict *le* to form *apple* or "
    "*lication* to form *application*."
)

st.sidebar.write("### How does it work?")
st.sidebar.caption(
    """
    The model is pre-trained on diverse datasets and fine-tuned with hyperparameters to generate contextually relevant text using embeddings and neural networks.

    """
)

st.sidebar.write("### Hyperparameters/ tuning knobs:")
st.sidebar.caption("""
 
    - Context Length
    - Embedding Size and MLP Architecture
    - Random seed
""")

st.sidebar.write("### Why are predictions dynamic?")
st.sidebar.caption(
    """
    The randomness in predictions is influenced by parameters like the random seed and the model's learned probabilities. 
    Varying seeds or hyperparameters can lead to diverse outputs for the same input. This flexibility is intentional, 
    showcasing the power of probabilistic text generation.
    """
)


choice = st.radio(
    "Do you wish to enter the initiating text?",
    ["Yes", "No"],
    index=None,
)

# Function to generate text
g = torch.Generator()
g.manual_seed(seed_n)
def generate_text(model, itos, stoi, block_size, max_len, start_str=None):
    context = [0] * block_size  

    if start_str is not None:
        for s in start_str:
            context = context[1:] + [stoi.get(s, 0)] 

    text = start_str if start_str else ""
    for i in range(max_len):
        x = torch.tensor(context).unsqueeze(0).to(device)
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        ch = itos.get(ix, itos[0])
        text += ch
        context = context[1:] + [ix]
    return text

# Writing the text on the screen
def write_text(str):
        for word in str.split(" "):
            yield word + " "
            time.sleep(0.03)

####################################################################################################################################################################################
##########################((((((((((((((((((((((((((((((((((((((((((((((((((((((((A SUITABLE BOY))))))))))))))))))))))))))))))))))))))))))))))))))))))))############################
if dataset == 'A Suitable Boy':
    file_path = r'D:\GITHUB\new_char\suitableboy.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    new_text = "".join([character.lower() for character in content])

    characters = sorted(list(set(''.join(new_text))))
    stoi = {s:i+1 for i, s in enumerate(characters)}
    stoi['`'] = 0 
    itos = {i: s for s, i in stoi.items()}

    model = NextChar(block_size=block_size, vocab_size=len(stoi), emb_dim=emb_dim, hidden_size=128).to(device)
    if emb_dim == 20:
        try:
            checkpoint = torch.load('suitableboy50_20_4002_128_64.pth', map_location=device)
            checkpoint = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()} 
            model.load_state_dict(checkpoint)
        except Exception as e:
            st.write(f"Error loading model: {e}")
    elif emb_dim == 40: 
        try:
            checkpoint = torch.load('suitableboy50_40_4002_128_64.pth', map_location=device)
            checkpoint = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()} 
            model.load_state_dict(checkpoint)
        except Exception as e:
            st.write(f"Error loading model: {e}")
    elif emb_dim == 80:
        try:
            checkpoint = torch.load('suitableboy80_80_4002_128_64.pth', map_location=device)
            checkpoint = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()} 
            model.load_state_dict(checkpoint)
        except Exception as e:
            st.write(f"Error loading model: {e}")

    if choice == "Yes":
        prompt = st.text_input("Enter your text here(less than 1000 charcters)")
        if prompt:
            st.write(f"You entered: {prompt}")
            if len(prompt) > 1000:
                prompt = prompt[:1000]
                for i in prompt:
                    if i not in characters:
                        prompt = prompt.replace(i, "")
            seed_text = prompt
            seed_text = seed_text.lower()
    elif choice == "No":
        l = st.slider("Select the length of the seed text", min_value=10, max_value=1000)
        st.write("Let's make seed text  random")
        start = np.random.randint(0,len(new_text) - l)
        end = start + l
        seed_text = new_text[start:end]

    else:
        st.write("Please make a selection")

    button = st.button("Generate text")
    if button:
        st.subheader("Seed text")
        st.write_stream(write_text(seed_text))

        generated_text = generate_text(model, itos, stoi, block_size, max_len, start_str=seed_text)
        st.subheader("Generated text:")
        st.write_stream(write_text(generated_text))

####################################################################################################################################################################################
################################((((((((((((((((((((((((((((((((((((((((((((((((LORD OF THE FLIES))))))))))))))))))))))))))))))))))))))))))))))))###################################    
elif dataset == 'Lord of the Flies':
    file_path = r'D:\GITHUB\new_char\LordoftheFlies.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    new_text = "".join([character.lower() for character in content])

    characters = sorted(list(set(''.join(new_text))))
    stoi = {s:i+1 for i, s in enumerate(characters)}
    stoi['`'] = 0 
    itos = {i: s for s, i in stoi.items()}

    model = NextChar(block_size=block_size, vocab_size=len(stoi), emb_dim=emb_dim, hidden_size=128).to(device)
    if emb_dim == 20:
        try:
            checkpoint = torch.load('lordoftheflies50_20_4002_128_64.pth', map_location=device)
            checkpoint = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()} 
            model.load_state_dict(checkpoint)
        except Exception as e:
            st.write(f"Error loading model: {e}")
    elif emb_dim == 40: 
        try:
            checkpoint = torch.load('lordoftheflies50_40_4002_128_64.pth', map_location=device)
            checkpoint = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()} 
            model.load_state_dict(checkpoint)
        except Exception as e:
            st.write(f"Error loading model: {e}")
    elif emb_dim == 80:
        try:
            checkpoint = torch.load('lordoftheflies80_80_4002_128_64.pth', map_location=device)
            checkpoint = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()} 
            model.load_state_dict(checkpoint)
        except Exception as e:
            st.write(f"Error loading model: {e}")

    if choice == "Yes":
        prompt = st.text_input("Enter your text here(less than 1000 charcters)")
        if prompt:
            st.write(f"You entered: {prompt}")
            if len(prompt) > 1000:
                prompt = prompt[:1000]
                for i in prompt:
                    if i not in characters:
                        prompt = prompt.replace(i, "")
            seed_text = prompt
            seed_text = seed_text.lower()
    elif choice == "No":
        l = st.slider("Select the length of the seed text", min_value=10, max_value=1000)
        st.write("Let's make seed text  random")
        start = np.random.randint(0,len(new_text) - l)
        end = start + l
        seed_text = new_text[start:end]

    else:
        st.write("Please make a selection")

    button = st.button("Generate text")
    if button:
        st.subheader("Seed text")
        st.write_stream(write_text(seed_text))

        generated_text = generate_text(model, itos, stoi, block_size, max_len, start_str=seed_text)
        st.subheader("Generated text:")
        st.write_stream(write_text(generated_text))

####################################################################################################################################################################################
################################((((((((((((((((((((((((((((((((((((((((((((((((MALGUDI DAYS))))))))))))))))))))))))))))))))))))))))))))))))###################################    
elif dataset == 'Malgudi Days':

    file_path = r'D:\GITHUB\new_char\MalgudiDays.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    new_text = "".join([character.lower() for character in content])

    characters = sorted(list(set(''.join(new_text))))
    stoi = {s:i+1 for i, s in enumerate(characters)}
    stoi['`'] = 0 
    itos = {i: s for s, i in stoi.items()}



    model = NextChar(block_size=block_size, vocab_size=len(stoi), emb_dim=emb_dim, hidden_size=128).to(device)
    if emb_dim == 20:
        try:
            checkpoint = torch.load('malgudidays50_20_4002_128_64.pth', map_location=device)
            checkpoint = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()} 
            model.load_state_dict(checkpoint)
        except Exception as e:
            st.write(f"Error loading model: {e}")
    elif emb_dim == 40: 
        try:
            checkpoint = torch.load('malgudidays50_40_4002_128_64.pth', map_location=device)
            checkpoint = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()} 
            model.load_state_dict(checkpoint)
        except Exception as e:
            st.write(f"Error loading model: {e}")
    elif emb_dim == 80:
        try:
            checkpoint = torch.load('malgudidays80_80_4002_128_64.pth', map_location=device)
            checkpoint = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()} 
            model.load_state_dict(checkpoint)
        except Exception as e:
            st.write(f"Error loading model: {e}")

    if choice == "Yes":
        prompt = st.text_input("Enter your text here(less than 1000 charcters)")
        if prompt:
            st.write(f"You entered: {prompt}")
            if len(prompt) > 1000:
                prompt = prompt[:1000]
                for i in prompt:
                    if i not in characters:
                        prompt = prompt.replace(i, "")
            seed_text = prompt
            seed_text = seed_text.lower()
    elif choice == "No":
        l = st.slider("Select the length of the seed text", min_value=10, max_value=1000)
        st.write("Let's make seed text  random")
        start = np.random.randint(0,len(new_text) - l)
        end = start + l
        seed_text = new_text[start:end]

    else:
        st.write("Please make a selection")

    button = st.button("Generate text")
    if button:
        st.subheader("Seed text")
        st.write_stream(write_text(seed_text))

        generated_text = generate_text(model, itos, stoi, block_size, max_len, start_str=seed_text)
        st.subheader("Generated text:")
        st.write_stream(write_text(generated_text))
####################################################################################################################################################################################
####################################################################################################################################################################################
################################((((((((((((((((((((((((((((((((((((((((((((((((THREE MEN IN A BOAT))))))))))))))))))))))))))))))))))))))))))))))))###################################    
elif dataset == 'Three Men In A Boat':

    file_path = r'D:\GITHUB\new_char\Threemeninaboat.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    new_text = "".join([character.lower() for character in content])

    characters = sorted(list(set(''.join(new_text))))
    stoi = {s:i+1 for i, s in enumerate(characters)}
    stoi['`'] = 0 
    itos = {i: s for s, i in stoi.items()}

    model = NextChar(block_size=block_size, vocab_size=len(stoi), emb_dim=emb_dim, hidden_size=128).to(device)
    if emb_dim == 20:
        try:
            checkpoint = torch.load('threemeninaboat50_20_4002_128_64.pth', map_location=device)
            checkpoint = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()} 
            model.load_state_dict(checkpoint)
        except Exception as e:
            st.write(f"Error loading model: {e}")
    elif emb_dim == 40: 
        try:
            checkpoint = torch.load('threemeninaboat50_40_4002_128_64.pth', map_location=device)
            checkpoint = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()} 
            model.load_state_dict(checkpoint)
        except Exception as e:
            st.write(f"Error loading model: {e}")
    elif emb_dim == 80:
        try:
            checkpoint = torch.load('threemeninaboat80_80_4002_128_64.pth', map_location=device)
            checkpoint = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()} 
            model.load_state_dict(checkpoint)
        except Exception as e:
            st.write(f"Error loading model: {e}")

    if choice == "Yes":
        prompt = st.text_input("Enter your text here(less than 1000 charcters)")
        if prompt:
            st.write(f"You entered: {prompt}")
            if len(prompt) > 1000:
                prompt = prompt[:1000]
                for i in prompt:
                    if i not in characters:
                        prompt = prompt.replace(i, "")
            seed_text = prompt
            seed_text = seed_text.lower()
    elif choice == "No":
        l = st.slider("Select the length of the seed text", min_value=10, max_value=1000)
        st.write("Let's make seed text  random")
        start = np.random.randint(0,len(new_text) - l)
        end = start + l
        seed_text = new_text[start:end]

    else:
        st.write("Please make a selection")

    button = st.button("Generate text")
    if button:
        st.subheader("Seed text")
        st.write_stream(write_text(seed_text))

        generated_text = generate_text(model, itos, stoi, block_size, max_len, start_str=seed_text)
        st.subheader("Generated text:")
        st.write_stream(write_text(generated_text))
####################################################################################################################################################################################
################################((((((((((((((((((((((((((((((((((((((((((((((((-----------------))))))))))))))))))))))))))))))))))))))))))))))))###################################    
# GitHub Link at the End
st.markdown(
    """
    ---
    Learn more https://github.com/your-github-link
    """,
    unsafe_allow_html=True
)

