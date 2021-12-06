import torch
from kospeech.model_builder import build_deepspeech2
from kospeech.data.audio import (
    FilterBankConfig,
    MelSpectrogramConfig,
    MfccConfig,
    SpectrogramConfig,
)
from kospeech.models import (
    DeepSpeech2Config,
    JointCTCAttentionLASConfig,
    ListenAttendSpellConfig,
    TransformerConfig,
    JointCTCAttentionTransformerConfig,
    JasperConfig,
    ConformerSmallConfig,
    ConformerMediumConfig,
    ConformerLargeConfig,
    RNNTransducerConfig,
)

config = DeepSpeech2Config
input_size = 80
len_vocab = 200
device = torch.device('cpu')
model = build_deepspeech2(
    input_size=input_size,
    num_classes=len_vocab,
    rnn_type='gru',
    num_rnn_layers=3,
    rnn_hidden_dim=1024,
    dropout_p=0.3,
    bidirectional=True,
    activation='hardtanh',
    device=device,
)
