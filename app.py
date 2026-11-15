import math
import numpy as np
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)
MOD_CLASSES = ['BPSK', 'QPSK', '8PSK', '16-QAM', '64-QAM']
NUM_SYMBOLS = 128


def generate_iq_sample(mod_type, snr_db):
    """Generate complex baseband symbols impaired by flat Rayleigh fading and AWGN."""
    if mod_type == 'BPSK':
        bits = np.random.randint(0, 2, NUM_SYMBOLS)
        symbols = (2 * bits - 1).astype(complex)
    elif mod_type == 'QPSK':
        idx = np.random.randint(0, 4, NUM_SYMBOLS)
        mapping = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        symbols = mapping[idx]
    elif mod_type == '8PSK':
        idx = np.random.randint(0, 8, NUM_SYMBOLS)
        symbols = np.exp(1j * idx * (2 * np.pi / 8))
    elif mod_type == '16-QAM':
        real = np.random.choice([-3, -1, 1, 3], NUM_SYMBOLS)
        imag = np.random.choice([-3, -1, 1, 3], NUM_SYMBOLS)
        symbols = (real + 1j * imag) / np.sqrt(10)
    else:
        real = np.random.choice([-7, -5, -3, -1, 1, 3, 5, 7], NUM_SYMBOLS)
        imag = np.random.choice([-7, -5, -3, -1, 1, 3, 5, 7], NUM_SYMBOLS)
        symbols = (real + 1j * imag) / np.sqrt(42)

    signal_power = float(np.mean(np.abs(symbols) ** 2))
    snr_linear = 10 ** (float(snr_db) / 10.0)
    noise_variance = signal_power / snr_linear
    noise = np.sqrt(noise_variance / 2) * (
        np.random.randn(NUM_SYMBOLS) + 1j * np.random.randn(NUM_SYMBOLS)
    )
    h = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
    return h * symbols + noise


def mock_deep_learning_classifier(rx_symbols, true_mod):
    """Placeholder confidence generator; replace with a trained model for real inference."""
    probs = np.random.dirichlet(np.ones(len(MOD_CLASSES)) * 0.5)
    true_idx = MOD_CLASSES.index(true_mod)
    probs[true_idx] += 4.0
    probs = np.maximum(probs, 0.01)
    probs /= np.sum(probs)
    return {MOD_CLASSES[i]: round(float(probs[i]) * 100, 2) for i in range(len(MOD_CLASSES))}


def q_function(x):
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def theoretical_ber(mod_type, ebn0_db):
    """Uncoded coherent AWGN BER approximation, used as a reference curve."""
    ebn0 = 10 ** (float(ebn0_db) / 10.0)
    if mod_type in ('BPSK', 'QPSK'):
        return 0.5 * math.erfc(math.sqrt(ebn0))
    if mod_type == '8PSK':
        k = 3
        return min(0.5, (2.0 / k) * q_function(math.sqrt(2 * k * ebn0) * math.sin(math.pi / 8)))
    m, k = (16, 4) if mod_type == '16-QAM' else (64, 6)
    return min(0.5, (4.0 / k) * (1 - 1 / math.sqrt(m)) *
               q_function(math.sqrt(3 * k * ebn0 / (m - 1))))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify_signal')
def classify_signal():
    mod_type = request.args.get('type', 'QPSK')
    if mod_type not in MOD_CLASSES:
        return jsonify({'error': 'Unsupported modulation format'}), 400
    try:
        snr = float(request.args.get('snr', 10.0))
    except ValueError:
        return jsonify({'error': 'SNR must be numeric'}), 400
    if not -2 <= snr <= 25:
        return jsonify({'error': 'SNR must be between -2 and 25 dB'}), 400

    iq_data = generate_iq_sample(mod_type, snr)
    confidence_scores = mock_deep_learning_classifier(iq_data, mod_type)

    # FFT power spectrum (normalized to its peak, in dB).
    spectrum = np.fft.fftshift(np.fft.fft(iq_data, n=512))
    power_db = 10 * np.log10(np.maximum(np.abs(spectrum) ** 2, 1e-12))
    power_db -= np.max(power_db)
    frequency = np.fft.fftshift(np.fft.fftfreq(512, d=1.0))

    return jsonify({
        'modulation': mod_type,
        'snr': snr,
        'i_coords': np.real(iq_data).tolist(),
        'q_coords': np.imag(iq_data).tolist(),
        'confidence': confidence_scores,
        'fft_frequency': frequency.tolist(),
        'fft_power_db': power_db.tolist(),
        'ber_awgn_reference': theoretical_ber(mod_type, snr),
        'ber_label': 'Theoretical uncoded AWGN reference (not measured BER)'
    })


@app.route('/ber_vs_snr')
def ber_vs_snr():
    mod_type = request.args.get('type', 'QPSK')
    if mod_type not in MOD_CLASSES:
        return jsonify({'error': 'Unsupported modulation format'}), 400
    snr_values = list(range(-2, 26))
    return jsonify({
        'modulation': mod_type,
        'snr_db': snr_values,
        'ber': [theoretical_ber(mod_type, value) for value in snr_values],
        'model': 'Uncoded theoretical AWGN approximation'
    })


if __name__ == '__main__':
    print('Starting SignalIQ RF analysis on http://127.0.0.1:5000')
    app.run(debug=True, port=5000)
