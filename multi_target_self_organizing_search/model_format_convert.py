import os.path as osp
import torch


def main():
    model_path = "./results/stp_actor_critic_4p_1t_6x6_reward_10_1e_4/stp_actor_critic_4p_1t_6x6_reward_10_1e_4_s0/pyt_save/"
    model_name = "model9999.pt"

    model = torch.load(osp.join(model_path, model_name))

    # Get parameters.
    torch.save(model.state_dict(), osp.join(model_path, "model.pth"))
    print("Saved PyTorch Model State to:", osp.join(model_path, "model.pth"))
    pass


if __name__ == "__main__":
    main()
