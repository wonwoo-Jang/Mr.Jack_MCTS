import torch
import torch.optim as optim
import torch.nn.functional as F
from env import Engine
from agent import DetectorAgent, JackAgent
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from utils import mk_next_dir

def run_episode(detector_agent, jack_agent, engine, epsilon=0):
    detector_states = []
    jack_states = []
    detector_states.append(engine.extract_state(mode='detector'))
    jack_states.append(engine.extract_state(mode='jack'))

    while not engine.end:
        if engine.round % 2 == 1:
            agent_order = [detector_agent, jack_agent, jack_agent, detector_agent]
        else:
            agent_order = [jack_agent, detector_agent, detector_agent, jack_agent]

        for agent in agent_order:
            # phase가 'done'이 될 때까지 agent가 행동 결정
            while engine.phase != 'done' and not engine.end:
                engine = agent.select_best_next_state(engine, epsilon=epsilon)
                detector_states.append(engine.extract_state(mode='detector'))
                jack_states.append(engine.extract_state(mode='jack'))

                if engine.end:
                    break
            
            if engine.end:
                break

            # 캐릭터 행동 이후 자동 처리 로직
            engine.step()

        if engine.end:
            engine.visualize_board()
            detector_reward = 1.0 if not engine.jack_win else -1.0
            jack_reward = -1 * detector_reward
            break

    return detector_states, jack_states, detector_reward, jack_reward

def compute_gae(detector_reward, jack_reward, detector_values, jack_values, gamma=0.99, lam=0.95):
    # GAE 계산을 위한 reward 리스트 생성
    T_det = len(detector_values)
    T_jack = len(jack_values)

    detector_rewards = [0.0] * T_det
    jack_rewards = [0.0] * T_jack
    detector_rewards[-1] = detector_reward
    jack_rewards[-1] = jack_reward

    detector_returns = [0.0] * T_det
    jack_returns = [0.0] * T_jack

    gae = 0.0
    for t in reversed(range(T_det)):
        next_value = detector_values[t + 1] if t + 1 < T_det else 0.0
        delta = detector_rewards[t] + gamma * next_value - detector_values[t]
        gae = delta + gamma * lam * gae
        detector_returns[t] = gae + detector_values[t]

    gae = 0.0
    for t in reversed(range(T_jack)):
        next_value = jack_values[t + 1] if t + 1 < T_jack else 0.0
        delta = jack_rewards[t] + gamma * next_value - jack_values[t]
        gae = delta + gamma * lam * gae
        jack_returns[t] = gae + jack_values[t]

    return torch.tensor(detector_returns, dtype=torch.float32), torch.tensor(jack_returns, dtype=torch.float32)

def train():
    # PPO Hyperparameters
    lr = 1e-4
    gamma = 0.99
    lam = 0.95
    clip_epsilon = 0.2
    epochs = 500
    steps_per_epoch = 200

    # Initialize Environment and Agent
    detector_agent = DetectorAgent()
    jack_agent = JackAgent()
    optimizer = optim.Adam(detector_agent.value_model.parameters(), lr=lr)

    # Create experiment setting
    save_dir = mk_next_dir(base_dir='runs', prefix='jack_detector_ppo')
    writer = SummaryWriter(log_dir=save_dir)
    global_step = 0
    best_det_loss = float('inf')
    best_jack_loss = float('inf')

    # PPO Training Loop
    for epoch in tqdm(range(epochs)):
        det_epoch_loss = 0
        jack_epoch_loss = 0
        epsilon = max(0.05, 0.2 * (0.995 ** epoch))

        for _ in range(steps_per_epoch):
            engine = Engine()
            detector_states, jack_states, detector_reward, jack_reward = run_episode(detector_agent, jack_agent, engine, epsilon)

            det_board_inputs = torch.cat([s[0] for s in detector_states], dim=0)
            det_misc_inputs = torch.cat([s[1] for s in detector_states], dim=0)
            jack_board_inputs = torch.cat([s[0] for s in jack_states], dim=0)
            jack_misc_inputs = torch.cat([s[1] for s in jack_states], dim=0)

            with torch.no_grad():
                detector_values = detector_agent.value_model(det_board_inputs, det_misc_inputs)
                jack_values = jack_agent.value_model(jack_board_inputs, jack_misc_inputs)

            gae_detector, gae_jack = compute_gae(detector_reward, jack_reward, detector_values, jack_values, gamma, lam)

            # PPO Critic update for detector
            predicted_det_values = detector_agent.value_model(det_board_inputs, det_misc_inputs).squeeze()
            loss_det = F.mse_loss(predicted_det_values, gae_detector.detach())

            optimizer.zero_grad()
            loss_det.backward()
            optimizer.step()

            # PPO Critic update for jack
            predicted_jack_values = jack_agent.value_model(jack_board_inputs, jack_misc_inputs).squeeze()
            loss_jack = F.mse_loss(predicted_jack_values, gae_jack.detach())

            optimizer.zero_grad()
            loss_jack.backward()
            optimizer.step()

            det_epoch_loss += loss_det.item()
            jack_epoch_loss += loss_jack.item()

            global_step += 1
            if global_step % 100:
                writer.add_scalar("StepLoss/Detector", loss_det.item(), global_step)
                writer.add_scalar("StepLoss/Jack", loss_jack.item(), global_step)

        det_avg_loss = det_epoch_loss / steps_per_epoch
        jack_avg_loss = jack_epoch_loss / steps_per_epoch

        writer.add_scalar("EpochLoss/Detector", det_avg_loss, epoch)
        writer.add_scalar("EpochLoss/Jack", jack_avg_loss, epoch)

        # print(f"Epoch {epoch + 1}/{epochs}, Detector_Loss: {det_avg_loss:.4f}")
        # print(f"Epoch {epoch + 1}/{epochs}, Jack_Loss: {jack_avg_loss:.4f}")

        # Save trained models
        if det_avg_loss < best_det_loss:
            best_det_loss = det_avg_loss
            torch.save(detector_agent.value_model.state_dict(), os.path.join(save_dir, 'ckpt', f'detector_value_model_{epoch}th_epoch.pth'))

        if jack_avg_loss < best_jack_loss:
            best_jack_loss = jack_avg_loss
            torch.save(detector_agent.value_model.state_dict(), os.path.join(save_dir, 'ckpt', f'jack_value_model_{epoch}th_epoch.pth'))

if __name__ == '__main__':
    engine = Engine()
    detector_agent = DetectorAgent()
    jack_agent = JackAgent()
    run_episode(detector_agent, jack_agent, engine)
    # train()