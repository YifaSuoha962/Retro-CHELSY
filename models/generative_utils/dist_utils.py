import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math


# borrow from onmt
def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


#=======================================================================================================================
# Aux functions & Sample techs
#=======================================================================================================================
def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs


# borrow from https://github1s.com/bojone/vae/blob/master/vae_keras_cnn_gs.py
def GumbelSoftmax(logits, tau=.8, noise=1e-20):
    eps = torch.rand(size=logits.shape, device=logits.device) # uniform distribution on the interval [0, 1)
    outputs = logits - torch.log(-torch.log(eps + noise) + noise)
    return torch.softmax(outputs / tau, -1)


# borrow from https://blog.evjang.com/2016/11/tutorial-categorical-variational.html
def sample_gumbel(shape, device, eps=1e-20):
  """Sample from Gumbel(0, 1)"""
  U = torch.rand(size=shape, device=device) # uniform distribution on the interval [0, 1)
  return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature=1.0):
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(logits.shape, logits.device)
  return torch.softmax( y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=1.0, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    y_hard = onehot_from_logits(y)
    y = (y_hard - y).detach() + y
  return y


#=======================================================================================================================
# Layers & Activations
#=======================================================================================================================
# BN in \mu & \sigma, refer from https://spaces.ac.cn/archives/7381
class Scaler(nn.Module):
    """特殊的scale层，基于PyTorch实现
    """
    def __init__(self, tau=0.5):
        super(Scaler, self).__init__()
        self.tau = tau
        self.scale = None

    def build(self, input):
        # 创建一个可训练的权重，形状为输入的最后一个维度
        self.scale = nn.Parameter(torch.zeros(input.shape[-1], device=input.device))

    def forward(self, inputs, mode='positive'):
        if self.scale is None:
            self.build(inputs)

        if mode == 'positive':
            scale = self.tau + (1 - self.tau) * torch.sigmoid(self.scale)
        else:
            scale = (1 - self.tau) * torch.sigmoid(-self.scale)

        return inputs * torch.sqrt(scale)

class NonLinear(nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation=None):
        super(NonLinear, self).__init__()

        self.activation = activation
        self.linear = nn.Linear(int(input_size), int(output_size), bias=bias)

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation( h )

        return h

class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, x):
        return torch.cat( F.relu(x), F.relu(-x), 1 )


#=======================================================================================================================
# Models
#=======================================================================================================================
class BD_HCVAE_mod(nn.Module):
    def __init__(self, args):
        super(BD_HCVAE_mod, self).__init__()
        self.args = args
        self.latent_size = self.args.conti_size + self.args.disc_size

        self.gamma = 0.5  # args.gamma
        self.delta_rate = 1

        # sample discrete reaction types
        self.prior_discr = NonLinear(self.args.decoder_hidden_size, self.args.disc_size, activation=nn.ELU())
        # self.recog_discr = NonLinear(self.args.decoder_hidden_size * 2, self.args.disc_size, activation=nn.ELU())

        # (original version) sample feature transformation within specific rxn type
        self.prior_conti = nn.GRU(input_size=self.args.decoder_hidden_size,
                                  hidden_size=self.args.decoder_hidden_size * 2,
                                  num_layers=1,
                                  batch_first=True,
                                  bidirectional=True)       # False in default
        self.recog_conti = nn.GRU(input_size=self.args.decoder_hidden_size,
                                  hidden_size=self.args.decoder_hidden_size * 2,
                                  num_layers=1,
                                  batch_first=True,
                                  bidirectional=True)       # False in default
        # batch_norm to alleviate kl collapse
        self.mu_bn = nn.BatchNorm1d(self.args.decoder_hidden_size)
        self.bn_layer = nn.BatchNorm1d(self.args.decoder_hidden_size)
        self.scaler = Scaler()

    # AUXILIARY METHODS
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def sample_prior(self, x):
        out_put, last_state = self.prior_conti(x, None)  # for GRU version
        z_p_mean, z_p_logvar = torch.chunk(last_state[0], chunks=2, dim=-1)
        z_p_r = self.reparameterize(z_p_mean, z_p_logvar)

        z_p_logp = 0
        return z_p_r, z_p_logp, z_p_mean, z_p_logvar

    def sample_recog(self, x, y):
        xy = torch.cat([x, y], dim=1)
        out_put, last_state = self.prior_conti(xy, None)  # for GRU version
        z_q_mean, z_q_logvar = torch.chunk(last_state[0], chunks=2, dim=-1)

        """bn+drop (manual)"""
        # if z_q_mean.shape[0] > 1:
        #     # bn
        #     z_q_mean = self.bn_layer(z_q_mean)
        #     z_q_mean = self.scaler(z_q_mean, mode='positive')
        #     # dropout on var^2
        #     z_q_std = z_q_logvar.exp_()
        #     z_q_std = torch.dropout(z_q_std, p=self.args.sigma_dropout, train=True)
        #     z_q_std += self.delta_rate * 1.0 / (2 * math.e * math.pi)
        #     z_q_logvar = torch.log(z_q_std)

        """from du-vae"""
        self.mu_bn.weight.requires_grad = True
        ss = torch.mean(self.mu_bn.weight.data ** 2) ** 0.5
        # if ss < self.gamma:
        self.mu_bn.weight.data = self.mu_bn.weight.data * self.gamma / ss
        if z_q_mean.shape[0] > 1:
            z_q_mean = self.mu_bn(z_q_mean)
        z_q_logvar = torch.log(torch.exp(z_q_logvar) + self.delta_rate * 1.0 / (2 * math.e * math.pi))

        z_q_r = self.reparameterize(z_q_mean, z_q_logvar)

        z_q_logp = 0
        return z_q_r, z_q_logp, z_q_mean, z_q_logvar

    def reparameterize_discrete(self, logits):
        """
        Samples from a gumbel-softmax distribution using the reparameterization
        trick.

        The sample_gumbel_softmax() argument should be unormalized log-probs
        -> apply softmax at the output of the encoder to make it
           prob and after take the log (or equivalently log_softmax)
        """
        # return self.sample_gumbel_softmax(logits)
        return gumbel_softmax(logits, hard=True)

    def sample_gumbel_softmax(self, logits, EPS=1e-10, hard=False):
        """
        Samples from a gumbel-softmax distribution using the reparameterization
        trick.
        https://blog.evjang.com/2016/11/tutorial-categorical-variational.html
        1.  Sample from Gumbel(0, 1)
        2. Draw a sample from the Gumbel-Softmax distribution
        Args
        ----------
        logits : torch.Tensor
           logits: [MB, disc_size] unnormalized log-probs -> apply softmax at the output of the encoder to make it
           prob an log (or equivalently log_softmax): def reparameterize_discrete
        hard: if True, take argmax, but differentiate w.r.t. soft sample y

        Returns:
        ----------
        [MB, disc_size] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
        """

        # Sample from Gumbel(0, 1)
        # torch.rand returns a tensor filled with rn from a uniform distr. on the interval [0, 1)
        unif = torch.rand(logits.size())
        if self.args.cuda:
            unif = unif.cuda()
        gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
        gumbel = Variable(gumbel)

        # Draw a sample from the Gumbel-Softmax distribution
        y = logits + gumbel
        ttt = self.args.temp

        # log_logits = torch.log(logits + EPS)
        # y = log_logits + gumbel
        gumbel_softmax_samples = F.softmax(y / ttt, dim=-1)

        if hard:
            gumbel_softmax_samples_max, _ = torch.max(gumbel_softmax_samples, dim=gumbel_softmax_samples.dim() - 1,
                                                      keepdim=True)

            gumbel_softmax_samples_hard = Variable(
                torch.eq(gumbel_softmax_samples_max.data, gumbel_softmax_samples.data).float()
            )
            gumbel_softmax_samples_hard_diff = gumbel_softmax_samples_hard - gumbel_softmax_samples
            gumbel_softmax_samples = gumbel_softmax_samples_hard_diff.detach() + gumbel_softmax_samples

        return gumbel_softmax_samples

    # used in training
    def forward(self, x, y):
        z_p_cont_r, z_p_logp, z_p_mean, z_p_logvar = self.sample_prior(x)
        z_q_cont_r, z_q_logp, z_q_mean, z_q_logvar = self.sample_recog(x, y)

        # reaction category sign
        # print(f"z_q_cont_r.shape = {z_q_cont_r.shape}")
        z_q_discr = self.prior_discr(z_q_cont_r)
        z_q_discr_r = self.reparameterize_discrete(z_q_discr)
        return z_q_discr_r, z_q_mean, z_q_logvar, z_p_mean, z_p_logvar

    def inference(self, x):
        # TODO: 考虑一下要不要加 mi
        z_p_cont_r, z_p_logp, z_p_mean, z_p_logvar = self.sample_prior(x)
        z_p_discr = self.prior_discr(z_p_cont_r)
        z_p_discr_r = self.reparameterize_discrete(z_p_discr)
        return z_p_discr_r

    def vanilla_KL(self, prior_logits, recog_logits, reduction='sum'):
        '''
        KL( q(z|x,y) || p(z|x) )
        '''
        prior_logits = F.log_softmax(prior_logits, dim=-1)
        recog_logits = F.softmax(recog_logits, dim=-1)
        KL_discr = F.kl_div(prior_logits, recog_logits, reduction=reduction)
        return KL_discr

    def KL_continuous(self, mean_q, log_var_q, mean_p, log_var_p, average=False, dim=None):
        '''
        return:
        KL( q || p )
        KL_cont =  (log_var_p / log_var_q + torch.exp(log_var_q) / torch.exp(log_var_p) + torch.pow(mean_p - mean_q, 2) / torch.exp(log_var_p))

        '''
        # # Matrix calculations
        # # Determinants of diagonal covariances pv, qv
        # dlog_var_p = log_var_q.prod()
        # dlog_var_q = log_var_q.prod(dim)
        # # Inverse of diagonal covariance var_q
        # inv_var_p = 1. / np.exp(log_var_p)
        # # Difference between means pm, qm
        # diff = mean_q - mean_p
        # KL_cont = (0.5 *
        #         ((dlog_var_p / dlog_var_q)  # log |\Sigma_p| / |\Sigma_q|
        #          + (inv_var_p * log_var_q).sum(dim)  # + tr(\Sigma_p^{-1} * \Sigma_q)
        #          + (diff * inv_var_p * diff).sum(dim)  # + (\mu_q-\mu_p)^T\Sigma_p^{-1}(\mu_q-\mu_p)
        #          - len(mean_q)))  # - D : size_z

        KL_cont = 0.5 * (log_var_p - log_var_q  # log s_p,i^2 / s_q_i^2
                         + torch.exp(log_var_q) / torch.exp(log_var_p)  # s_q_i^2 / s_p_i^2
                         + torch.pow(mean_q - mean_p, 2) / torch.exp(log_var_p)  # (m_p_i - m_q_i)^2 / s_p_i^2
                         - 1)  # dim_z -> after sum D

        if average:
            KL_cont = torch.mean(KL_cont)
        else:
            KL_cont = torch.sum(KL_cont)
        return KL_cont

    def calculate_loss(self, x, y, beta=1., average=False):
        z_q_discr_r, z_q_mean, z_q_logvar, z_p_mean, z_p_logvar = self.forward(x, y)
        KL_discr = 0
        # KL(p(z|x,y,c) || q(z|x,c))
        KL_cont = self.KL_continuous(z_q_mean, z_q_logvar, z_p_mean, z_p_logvar, dim=-1)
        KL_loss = KL_discr + KL_cont

        return z_q_discr_r, KL_loss

    def calculate_mi(self, x, z_p_mean, z_p_logvar):
        """Approximate the mutual information between x and z
            I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))
            Returns: Float
        """
        # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)
        neg_entropy = (-0.5 * z_p_mean.shape[1] * math.log(2 * math.pi) - 0.5 * (1 + z_p_logvar).sum(-1)).mean()
        # z: [[1, x_batch, nz]]
        z_p_cont_r = self.reparameterize(z_p_mean, z_p_logvar).unsqueeze(1)
        # mu_z|x, sigma_z|x: [1, x_batch, nz]
        z_p_mean, z_p_logvar = z_p_mean.unsqueeze(0), z_p_logvar.unsqueeze(0)
        z_p_var = z_p_logvar.exp()
        # deviation
        dev = z_p_cont_r - z_p_mean

        log_density = -0.5 * ((dev ** 2) / z_p_var).sum(dim=-1) - \
                      0.5 * (z_p_mean.shape[1] * math.log(2 * math.pi) + z_p_logvar.sum(-1))
        # log q(z): aggregate posterior
        # [z_batch]
        log_qz = log_sum_exp(log_density, dim=1) - math.log(z_p_mean.shape[0])

        return (neg_entropy - log_qz.mean(-1)).item()

    def sample_categorical(self, N, dim):
        samples = torch.zeros((N, dim))
        samples[np.arange(N), np.random.randint(0, dim, N)] = 1.
        if self.args.cuda:
            samples = samples.cuda
        return torch.Tensor(samples)

