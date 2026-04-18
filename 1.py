import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import normalize

model = SentenceTransformer('intfloat/e5-large-v2')

text1 = "Клиент заинтересован в аренде склада"
text2 = "Собака пошла гулять"

emb1 = model.encode(text1, convert_to_tensor=True)
emb2 = model.encode(text2, convert_to_tensor=True)

emb1 = emb1 /emb1.norm()
emb2 /= emb2 / emb2.norm()

similarity = util.cos_sim(emb1, emb2).item()

print(similarity)
