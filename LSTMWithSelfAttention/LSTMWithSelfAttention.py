class LSTMWithSelfAttention(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=3, output_size=2):
        super(LSTMWithSelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)

        #각각의 대한 projection이 모델이 시점 간 의미 있는 관계를 더 정밀하고 유연하게 학습할 수 있게 해준다.
        self.query_proj = nn.Linear(hidden_size, hidden_size)  #LSTM 출력 → Query로 변환(내가 보고 싶은 것(Q))
        self.key_proj = nn.Linear(hidden_size, hidden_size)  #LSTM 출력 → Key로 변환(상대가 가진 정보(K))
        self.value_proj = nn.Linear(hidden_size, hidden_size)  #LSTM 출력 → Value로 변환(그 정보의 실제 값(V))

        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        x, _ = self.lstm(x, (h0, c0))  # x = (Batch_size, Time-step, Hidden_size) -> 모든 시점에서의 각각의 은닉층
        # last_hidden = hn[-1] -> self-attention은 최종 은닉층을 사용하지 않고 전체 은닉층을 사용

        Q = self.query_proj(x)  #내가 보고 싶은 것(Q) -> (Batch_size, Time-step, Hidden_size)
        K = self.key_proj(x)  #상대가 가진 정보(K) -> (Batch_size, Time-step, Hidden_size)
        V = self.value_proj(x)  #그 정보의 실제 값(V) -> (Batch_size, Time-step, Hidden_size)
        
        #Self Attention 메커니즘 방식 ->  Self-Attention은 각 시점별로 attention을 계산하는 방식
        #1. 모든 시점에서의 은닉층 (내적) 모든 시점에서의 은닉층 -> 모든 시점에서의 각 은닉 상태와 모든 시점에서의 각 은닉 상태의 유사도를 스코어로 계산, 중요한 시점에 더 높은 가중치를 주어 정보를 요약
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_size**0.5)  #(B, T, H) × (B, H, T) → (B, T, T) -> **중요** scores[b, i, j]는 i번째 시점의 Query가 j번째 시점의 Key와 얼마나 유사한지를 의미 **
        
        #2. 스코어를 소프트맥스에 거치게 하여 총합 1로 만듦, softmax로 정규화된 attention weight를 통해 "중요한 시점"을 강조
        attention_weight = torch.softmax(scores, dim=-1)  #(batch_size, Time_step, Time_step)

        #3. 유사도 스코어 (내적) 모든 시점에서의 은닉층 -> 모든 은닉 상태와 가중합(Weighted sum)하여 하나의 요약 벡터(context)를 생성
        context = torch.matmul(attention_weight, V) #최종 결과 출력 -> (Batch_size, Time-step, Hidden_size)
        #인코더의 모든 문맥을 포함하고 있다하여 context라고 부름

        pooled = torch.mean(context, dim=1) #전체 시퀀스를 대표하는 단 하나의 벡터가 필요하기 때문에 시퀀스 차원 T에 대해 전류값 평균을 내서, 각 윈도우마다 대표적인 정보 하나만 뽑는 방식 → shape: (B, H)
        #또한 MLP는 입력으로 (B, H)를 받기 때문에 (B, T, H)는 넣어줄 수 없음
        #또한 mean pooling 말고도 max pooling 등등이 있음

        #MLP
        features = self.relu(self.fc1(pooled))
        features = self.fc2(features)

        return features
