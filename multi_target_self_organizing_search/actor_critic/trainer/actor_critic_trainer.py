from torch.optim import Adam

from .experience_buffer import ExperienceBuffer

from actor_critic.common.mpi_pytorch import mpi_avg_grads, syn_params
from actor_critic.common.models import MLPActorCritic


class ActorCriticAgents:
    def __init__(self, n_agents, obs_dim, act_dim, steps_per_epoch, gamma, lam, pi_lr, vf_lr, train_v_iters):
        self.n_agents = n_agents
        self.buffer = ExperienceBuffer(obs_dim, (), steps_per_epoch, n_agents, gamma, lam)
        self.data = None

        self.model = MLPActorCritic(obs_dim, act_dim, hidden_sizes=(400, 300))
        self.pi_optimizer = Adam(self.model.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.model.v.parameters(), lr=vf_lr)
        self.train_v_iters = train_v_iters

    def syn_params(self):
        syn_params(self.model)

    def update(self):
        self.data = self.buffer.get()

        # Get loss and info values before update.
        loss_pi_old, pi_info_old = self.compute_loss_pi()
        loss_pi_old = loss_pi_old.item()
        loss_v_old = self.compute_loss_v()

        # Train policy with a single step of gradient descent.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi()
        loss_pi.backward()
        # Average grads across MPI processes.
        mpi_avg_grads(self.model.pi)
        self.pi_optimizer.step()

        # Value function learning.
        loss_v = loss_v_old
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v()
            loss_v.backward()
            # Average grads across MPI processes.
            mpi_avg_grads(self.model.v)
            self.vf_optimizer.step()

        # Log changes from update.
        delta_loss_pi = loss_pi.item() - loss_pi_old
        delta_loss_v = loss_v.item() - loss_v_old
        kl, entropy = pi_info['kl'], pi_info['ent']
        return loss_pi_old, loss_v_old, delta_loss_pi, delta_loss_v, kl, entropy

    def compute_loss_pi(self):
        """
        Set up function for computing VPG policy loss.
        """
        obs, act, adv, logp_old = self.data['obs'], self.data['act'], self.data['adv'], self.data['logp']

        # Policy loss.
        pi, logp = self.model.pi(obs, act)
        loss_pi = -(logp * adv).mean()

        # Useful extra info.
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)

        return loss_pi, pi_info

    def compute_loss_v(self):
        """
        Set up function for computing value loss.
        """
        obs, ret = self.data['obs'], self.data['ret']
        loss_v = ((self.model.v(obs) - ret) ** 2).mean()
        return loss_v

