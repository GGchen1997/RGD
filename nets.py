import math

from torch import nn


from utils import *

from lib.sdes import VariancePreservingSDE, ScorePluginReverseSDE
import torch.nn.functional as F


class Swish(nn.Module):

    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.sigmoid(x) * x

mysoftplus = torch.nn.Softplus()
class SimpleMLP(nn.Module):

    def __init__(self,
                 in_dim,
                 hid_dim=1024,
                 out_dim=2):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )

        self.min_logstd = torch.log(torch.Tensor([0.1]))
        self.max_logstd = torch.log(torch.Tensor([0.2]))

    def forward(self, x):
        output = self.main(x)
        #(N, 2)
        mu, logstd = output[:, 0], output[:, 1]
        self.min_logstd = self.min_logstd.to(x.device)
        self.max_logstd = self.max_logstd.to(x.device)
        logstd = self.max_logstd - mysoftplus(self.max_logstd - logstd)
        logstd = self.min_logstd + mysoftplus(logstd - self.min_logstd)
        return mu, torch.exp(logstd)

class ModelEnsemble:
    def __init__(self, seeds, task_name, input_dim, device):
        self.models = []
        for seed in seeds:
            classifier = SimpleMLP(input_dim).to(device)
            state_dict = torch.load("model/" + task_name + "_proxy_" + str(seed) + ".pt")
            classifier.load_state_dict(state_dict)
            self.models.append(classifier)
    def predict(self, x):
        with torch.no_grad():
            predictions = [model(x)[0].squeeze() for model in self.models]
            mean_prediction = torch.mean(torch.stack(predictions), dim=0)
        return mean_prediction



def get_sinusoidal_positional_embedding(timesteps: torch.LongTensor, embedding_dim: int):
    """
    Copied and modified from
        https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/nn.py#L90

    From Fairseq in
        https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py#L15
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.size()) == 1
    timesteps = timesteps.to(torch.get_default_dtype())
    device = timesteps.device

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # bsz x embd
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), "constant", 0)
    assert list(emb.size()) == [timesteps.size(0), embedding_dim]
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim=256, hidden_dim=1024, output_dim=1024):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, temp):
        temb = get_sinusoidal_positional_embedding(temp, self.embedding_dim)
        temb = self.main(temb)
        return temb

class MLP(nn.Module):

    def __init__(
            self,
            input_dim=2,
            index_dim=1,
            hidden_dim=1024
    ):
        super().__init__()
        self.input_dim = input_dim
        self.index_dim = index_dim
        self.hidden_dim = hidden_dim
        self.y_dim = 1

        self.fc_x = nn.Linear(input_dim, hidden_dim)
        self.fc_y = TimestepEmbedding(output_dim=1024)
        self.fc_t = TimestepEmbedding(output_dim=1024)

        self.main = nn.Sequential(
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, input, t, y):
        # init
        sz = input.size()
        input = input.view(-1, self.input_dim)

        emb_x = self.fc_x(input)
        emb_y = self.fc_y(y.squeeze())
        emb_t = self.fc_t(t.squeeze())
        h = emb_y * emb_x + emb_t
        #print(emb_y.shape, emb_x.shape, emb_t.shape, h.shape)

        output = self.main(h)  # forward
        return output.view(*sz)

class DiffusionScore(nn.Module):

    def __init__(
            self,
            task_x,
            task_y,
            hidden_size=1024,
            beta_min=0.01,
            beta_max=2.0,
            dropout_p=0.15,
            simple_clip=True,
            T0 = 1.0,
            debias=False,
            vtype='rademacher'):
        super().__init__()

        self.dim_x = task_x.shape[-1]
        self.dim_y = task_y.shape[-1]

        self.task_y = task_y
        self.device = task_x.device

        self.clip_min = task_x.clone().detach().min(axis=0)[0].to(task_x.device)
        self.clip_max = task_x.clone().detach().max(axis=0)[0].to(task_x.device)

        self.beta_min = beta_min
        self.beta_max = beta_max

        self.T0 = T0
        self.vtype = vtype
        self.dropout_p = dropout_p
        self.simple_clip = simple_clip
        self.debias = debias

        self.score_estimator = MLP(input_dim=self.dim_x,
                                   index_dim=1,
                                   hidden_dim=hidden_size)
        #self.score_estimator = UNet_MLP(self.dim_x)
        self.T = torch.FloatTensor([self.T0]).to(task_x.device)
        #inference
        self.inf_sde = VariancePreservingSDE(beta_min=self.beta_min,
                                             beta_max=self.beta_max,
                                             T=self.T)
        #generation
        self.gen_sde = ScorePluginReverseSDE(self.inf_sde,
                                             self.score_estimator,
                                             self.T,
                                             vtype=self.vtype,
                                             debias=self.debias)


    def training_step(self, batch):
        x, y, w = batch

        if self.dropout_p:
            rand_mask = torch.rand(y.size())
            mask = (rand_mask <= self.dropout_p)
            y[mask] = 0.

        loss = self.gen_sde.dsm_weighted(
            x, y, w,
            clip=self.simple_clip,
            c_min=self.clip_min,
            c_max=self.clip_max).mean()  # forward and compute loss

        return loss

    def validation_step_elbo(self, batch):
        x, y = batch
        elbo = self.gen_sde.elbo_random_t_slice(x, y)
        return elbo

    def validation_step_agree(self, ensemble):
        N = 100
        iterations = 10
        agreement = 0
        for i in range(iterations):
            indexs = torch.randperm(self.task_y.shape[0])
            input_task_y = self.task_y[indexs][0:N].squeeze()

            x_0 = torch.randn(N, self.dim_x, device=self.device)
            x_s = heun_sampler_ode(self, x_0, ya=input_task_y,
                                   classifier=None, gamma_learn=False, selection=False)
            pred_score = ensemble.predict(x_s).squeeze()
            agreement = agreement - torch.mean(torch.pow(pred_score - input_task_y, 2))
        return agreement/iterations

