# DigitalOcean Deployment

Deploy both paper bots on one Ubuntu 24.04 LTS DigitalOcean Basic droplet.

## 1. Create the droplet

1. Create or sign in to a DigitalOcean account.
2. Create a Droplet.
3. Choose Ubuntu 24.04 LTS.
4. Choose Basic Regular, 1 GB RAM, 1 vCPU, 25 GB SSD.
5. Choose New York region.
6. Add SSH key authentication. On the laptop, create a key if needed:

```bash
ssh-keygen -t ed25519 -C "sid-tradingbot-do"
cat ~/.ssh/id_ed25519.pub
```

7. Paste the public key into DigitalOcean and create the droplet.
8. Connect:

```bash
ssh root@DROPLET_IP
```

## 2. Server environment

```bash
adduser --disabled-password --gecos "" trader
usermod -aG sudo trader
apt update && apt -y upgrade
apt install -y git python3 python3-venv python3-pip
sudo -iu trader
git clone https://github.com/sidrianchan/trading-bot.git ~/trading-bot
cd ~/trading-bot
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

If the GitHub remote is unavailable, copy the repo directory to `~/trading-bot` with `rsync` from the laptop.

## 3. Secrets

Create `/home/trader/trading-bot/.env`:

```bash
cat > /home/trader/trading-bot/.env <<'EOF'
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_PAPER=true
EOF
chmod 600 /home/trader/trading-bot/.env
```

Never commit `.env`. The repo ignores `.env`, `.kill_switch`, `logs/`, and `models/`.

## 4. systemd services

```bash
sudo cp /home/trader/trading-bot/deploy/systemd/trading-etf.service /etc/systemd/system/trading-etf.service
sudo cp /home/trader/trading-bot/deploy/systemd/trading-crypto.service /etc/systemd/system/trading-crypto.service
sudo cp /home/trader/trading-bot/deploy/systemd/trading-health.service /etc/systemd/system/trading-health.service
sudo cp /home/trader/trading-bot/deploy/systemd/trading-health.timer /etc/systemd/system/trading-health.timer
sudo systemctl daemon-reload
sudo systemctl enable --now trading-etf trading-crypto trading-health.timer
```

## 5. Verify

```bash
systemctl status trading-etf trading-crypto trading-health.timer
journalctl -u trading-etf -f
journalctl -u trading-crypto -f
cd /home/trader/trading-bot
. .venv/bin/activate
python main.py status --bot etf
python main.py status --bot crypto
```

Anytime status from the laptop:

```bash
ssh trader@DROPLET_IP 'cd ~/trading-bot && systemctl --no-pager status trading-etf trading-crypto'
```

## 6. Stop local Mac launchd after droplet is confirmed

```bash
launchctl bootout gui/$(id -u) /Users/sid/Library/LaunchAgents/com.tradingbot.paper.plist
launchctl print gui/$(id -u)/com.tradingbot.paper
```

The `launchctl print` command should fail or show the service is not loaded. After that, the droplet is the canonical running instance.
