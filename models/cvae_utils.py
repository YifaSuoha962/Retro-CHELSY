import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.optim import Optimizer
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
# from CGlow_modules import modules
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler



def gaussian_kld(recog_mu, recog_std, prior_mu, prior_std):
    kld = -0.5 * torch.sum(torch.log(torch.div(torch.pow(recog_std, 2), torch.pow(prior_std, 2)) + 1e-6)
                           - torch.div(torch.pow(recog_std, 2), torch.pow(prior_std, 2))
                           - torch.div(torch.pow(recog_mu - prior_mu, 2), torch.pow(prior_std, 2)) + 1, dim=-1)
    return kld


def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs


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


# borrow from https://github1s.com/bojone/vae/blob/master/vae_keras_cnn_gs.py
def GumbelSoftmax(logits, tau=.8, noise=1e-20):
    eps = torch.rand(size=logits.shape, device=logits.device)  # uniform distribution on the interval [0, 1)
    outputs = logits - torch.log(-torch.log(eps + noise) + noise)
    return torch.softmax(outputs / tau, -1)


# borrow from https://blog.evjang.com/2016/11/tutorial-categorical-variational.html
def sample_gumbel(shape, device, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(size=shape, device=device)  # uniform distribution on the interval [0, 1)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1.0):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, logits.device)
    return torch.softmax(y / temperature, dim=-1)


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
    # plot_multinomial_dist(y[0].cpu().numpy())
    # print(f"y.shape[0] = {y.shape[0]}")
    # assert 1 == 2

    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y


def plot_multinomial_dist(z_p_disc):
    # 1D distribution curve
    demo_disc_dist = z_p_disc[0]
    plt.figure()
    plt.plot(z_p_disc, color='orange')
    x_major_locator = MultipleLocator(10)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.savefig('demo_cat_dist.png')


def plot_tsne(z_p_r, z_p_discr):
    # 2D distribution: tsne -> KDE

    z_scaler = StandardScaler()
    z_p_r_scaled = z_scaler.fit_transform(z_p_r)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(z_p_r_scaled)
    plt.figure(figsize=(16, 10))

    tsne_df = pd.DataFrame(data=tsne_results, columns=["tsne-2d-one", "tsne-2d-two"])
    tsne_df['y'] = z_p_discr.argmax(-1)  # 添加分类标签
    dups = tsne_df['y'].value_counts()
    tsne_df['dups'] = tsne_df['y'].map(dups)

    print(f"z_p_r.shape = {z_p_r.shape}, z_p_discr.shape = {z_p_discr.shape}")
    print(f"tsne_df.head(3) = \n{tsne_df.head(3)}")
    print(f"dup_ys = {dups}, with shape = {dups.shape}, dups[24] = {dups[24]}")
    fig = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=tsne_df[tsne_df['dups'] >= 9],
        legend="full",
        alpha=0.3
    )
    fig.set_xlim(-20, 20)
    fig.set_ylim(-20, 20)
    plt.xlabel('Variable X')
    plt.ylabel('Variable Y')
    plt.title('2D Kernel density estimation')
    plt.savefig(f'bdcvae-mod-tsne_sctterplot.png')

    # assert 1 == 2

    # sns.kdeplot(tsne_df["tsne-2d-one"], tsne_df["tsne-2d-two"], cmap='coolwarm')

    # plt.savefig(f'bdcvae-mod-tsne_kdeplt.png')


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


# vanilla DCVAE
class DCVAE(nn.Module):
    def __init__(self, disc_size, decoder_hidden_size, temp):
        super(DCVAE, self).__init__()
        self.decoder_hidden_size = decoder_hidden_size
        self.prior_net = nn.GRU(input_size=self.decoder_hidden_size,
                                  hidden_size=disc_size,
                                  num_layers=1,
                                  batch_first=True,
                                  bidirectional=False)
        self.recog_net = nn.GRU(input_size=self.decoder_hidden_size,
                                  hidden_size=disc_size,
                                  num_layers=1,
                                  batch_first=True,
                                  bidirectional=False)

        self.temp = temp


    def forward(self, x, y):

        logits = self.prior_net(x, None)[-1][0]
        prior = F.log_softmax(logits, dim=-1)

        logits = self.recog_net(torch.cat([x, y], dim=1), None)[-1][0]  # GRU
        recog = F.softmax(logits, dim=-1)

        kld_loss = F.kl_div(prior, recog, reduction="sum")
        c_sample = gumbel_softmax(logits, hard=True)

        return c_sample, kld_loss

    def calculate_loss(self, x, y):
        return self.forward(x, y)

    def inference(self, x):
        logits = self.prior_net(x, None)[-1][0]
        c_sample = gumbel_softmax(logits, hard=True)

        return c_sample


##################################################################################
# Modified HCVAE with BN & Dropout
##################################################################################

class BD_HCVAE_mod(nn.Module):
    def __init__(self, conti_size, disc_size, decoder_hidden_size, temp):
        super(BD_HCVAE_mod, self).__init__()
        self.conti_size = conti_size
        self.disc_size = disc_size
        self.decoder_hidden_size = decoder_hidden_size

        self.gamma = 0.5  # args.gamma
        self.delta_rate = 1

        # sample discrete reaction types
        self.prior_discr = NonLinear(self.decoder_hidden_size, self.disc_size, activation=nn.ELU())
        self.recog_discr = NonLinear(self.decoder_hidden_size, self.disc_size, activation=nn.ELU())

        # (original version) sample feature transformation within specific rxn type
        self.prior_conti = nn.GRU(input_size=self.decoder_hidden_size,
                                  hidden_size=self.decoder_hidden_size * 2,
                                  num_layers=1,
                                  batch_first=True,
                                  bidirectional=False)
        self.recog_conti = nn.GRU(input_size=self.decoder_hidden_size,
                                  hidden_size=self.decoder_hidden_size * 2,
                                  num_layers=1,
                                  batch_first=True,
                                  bidirectional=False)
        # batch_norm to alleviate kl collapse
        self.mu_bn = nn.BatchNorm1d(self.decoder_hidden_size)
        self.bn_layer = nn.BatchNorm1d(self.decoder_hidden_size)
        self.scaler = Scaler()

        self.temp = temp

    # AUXILIARY METHODS
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()

        eps = torch.cuda.FloatTensor(std.size()).normal_()

        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def sample_prior(self, x):
        out_put, last_state = self.prior_conti(x, None)  # for GRU version
        z_p_mean, z_p_logvar = torch.chunk(last_state[0], chunks=2, dim=-1)
        z_p_r = self.reparameterize(z_p_mean, z_p_logvar)

        # plot_kde_dist(z_p_r.cpu().numpy())
        # assert 1 == 2

        z_p_logp = 0
        return z_p_r, z_p_logp, z_p_mean, z_p_logvar

    def sample_recog(self, x, y):
        xy = torch.cat([x, y], dim=1)
        out_put, last_state = self.recog_conti(xy, None)  # for GRU version
        z_q_mean, z_q_logvar = torch.chunk(last_state[0], chunks=2, dim=-1)

        # if z_q_mean.shape[0] > 1:
        #     # bn
        #     z_q_mean = self.bn_layer(z_q_mean)
        #     z_q_mean = self.scaler(z_q_mean, mode='positive')
        #     # dropout on var^2
        #     z_q_std = z_q_logvar.exp_()
        #     z_q_std = torch.dropout(z_q_std, p=self.args.sigma_dropout, train=True)
        #     z_q_std += self.delta_rate * 1.0 / (2 * math.e * math.pi)
        #     z_q_logvar = torch.log(z_q_std)

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

        unif = unif.cuda()
        gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
        gumbel = Variable(gumbel)

        # Draw a sample from the Gumbel-Softmax distribution
        y = logits + gumbel
        ttt = self.temp

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

        # batch norm & dropout
        # if z_q_mean.shape[0] > 1:
        #     # bn
        #     z_q_mean = self.bn_layer(z_q_mean)
        #     z_q_mean = self.scaler(z_q_mean, mode='positive')
        #     # dropout on var^2
        #     # z_q_std = z_q_logvar.exp_()
        #     z_q_std = z_q_logvar.exp()
        #     z_q_std = torch.dropout(z_q_std, p=self.args.sigma_dropout, train=True)
        #     z_q_std += self.delta_rate * 1.0 / (2 * math.e * math.pi)
        #     z_q_logvar = torch.log(z_q_std)

        # reaction category sign
        # print(f"z_q_cont_r.shape = {z_q_cont_r.shape}")
        z_q_discr = self.recog_discr(z_q_cont_r)
        z_q_discr_r = self.reparameterize_discrete(z_q_discr)

        z_p_discr = self.prior_discr(z_p_cont_r)

        # z_q = torch.cat([z_q_cont_r, z_q_discr_r], dim=-1)
        return z_q_discr_r, z_q_mean, z_q_logvar, z_p_mean, z_p_logvar, z_q_discr, z_p_discr

    def inference(self, x):
        # TODO: 考虑一下要不要加 mi
        z_p_cont_r, z_p_logp, z_p_mean, z_p_logvar = self.sample_prior(x)
        z_p_discr = self.prior_discr(z_p_cont_r)
        z_p_discr_r = self.reparameterize_discrete(z_p_discr)

        # plot_tsne(z_p_cont_r.cpu().numpy(), z_p_discr_r.cpu().numpy())

        # z_p = torch.cat([z_p_cont_r, z_p_discr_r], -1)

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
        z_q_discr_r, z_q_mean, z_q_logvar, z_p_mean, z_p_logvar, z_q_discr, z_p_discr = self.forward(x, y)
        # KL(p(c|x,y) || q(c|x))
        KL_discr = self.vanilla_KL(z_p_discr, z_q_discr)
        # KL_discr = 0
        # KL(p(z|x,y,c) || q(z|x,c))
        KL_cont = self.KL_continuous(z_q_mean, z_q_logvar, z_p_mean, z_p_logvar, dim=-1)
        KL_loss = KL_discr + KL_cont

        return z_q_discr_r, KL_loss



def calculate_lower_bound(X_full, MB=100):
    '''
    X_full: the whole xtest or train dataset
    MB: size of MB
    return: ELBO
    I: nuber of MBs
    '''

    # X_full = self.prior_encoder(X_full, None)[-1][0]
    # CALCULATE LOWER BOUND:
    lower_bound = 0.
    KL_cont_all = 0.
    KL_discr_all = 0.

    # ceil(): returns ceiling value of x - the smallest integer not less than x
    I = int(math.ceil(X_full.size(0) / MB))
    # for i in range(I):
    # x = X_full[i * MB: (i + 1) * MB].view(-1, np.prod(self.args.decoder_hidden_size))
    # _, KLs = self.calculate_loss(x, average=True)
    # KL_cont_all += KL_cont.cpu().data[0]
    # KL_discr_all += KL_discr.cpu().data[0]

    lower_bound /= I
    return lower_bound


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


def log_Normal_diag(x, mean, log_var, average=False, dim=None):
    log_normal = -0.5 * (log_var + torch.pow(x - mean, 2) / torch.exp(log_var))
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)


def log_Normal_standard(x, average=False, dim=None):
    log_normal = -0.5 * torch.pow(x, 2)
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)


min_epsilon = 1e-5
max_epsilon = 1. - 1e-5


def log_Bernoulli(x, mean, average=False, dim=None):
    probs = torch.clamp(mean, min=min_epsilon, max=max_epsilon)
    log_bernoulli = x * torch.log(probs) + (1. - x) * torch.log(1. - probs)
    if average:
        return torch.mean(log_bernoulli, dim)
    else:
        return torch.sum(log_bernoulli, dim)


# =======================================================================================================================
# WEIGHTS INITS
# =======================================================================================================================
def xavier_init(m):
    s = np.sqrt(2. / (m.in_features + m.out_features))
    m.weight.data.normal_(0, s)


# =======================================================================================================================
def he_init(m):
    s = np.sqrt(2. / m.in_features)
    m.weight.data.normal_(0, s)


# =======================================================================================================================
def normal_init(m, mean=0., std=0.01):
    m.weight.data.normal_(mean, std)


# =======================================================================================================================

# =======================================================================================================================
# ACTIVATIONS
# =======================================================================================================================
class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, x):
        return torch.cat(F.relu(x), F.relu(-x), 1)


# =======================================================================================================================
# LAYERS
# =======================================================================================================================
class NonLinear(nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation=None):
        super(NonLinear, self).__init__()

        self.activation = activation
        self.linear = nn.Linear(int(input_size), int(output_size), bias=bias)

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)

        return h


# =======================================================================================================================
class GatedDense(nn.Module):
    def __init__(self, input_size, output_size, activation=None):
        super(GatedDense, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.Linear(input_size, output_size)
        self.g = nn.Linear(input_size, output_size)

    def forward(self, x):
        h = self.h(x)
        if self.activation is not None:
            h = self.activation(self.h(x))

        # g = F.relu(self.g(x))
        g = self.sigmoid(self.g(x))

        return h * g


# =======================================================================================================================
class AdamNormGrad(Optimizer):
    """Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(AdamNormGrad, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the models
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                #############################################
                # normalize grdients
                grad = grad / (torch.norm(grad, 2) + 1.e-7)
                #############################################

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


