import argparse
import base64
import numpy as np
import pandas as pd
import seaborn as sns


from io import BytesIO
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROLLERS, CONTROL_START_IDX

from scipy.optimize import minimize, LinearConstraint, OptimizeResult

sns.set_theme()
SAMPLE_ROLLOUTS = 5


def img2base64(fig):
  buf = BytesIO()
  fig.savefig(buf, format='png')
  data = base64.b64encode(buf.getbuffer()).decode("ascii")
  return data


def create_report(test, baseline, sample_rollouts, costs):
  res = []
  res.append("<h1>Comma Controls Challenge: Report</h1>")
  res.append(f"<b>Test Controller: {test}, Baseline Controller: {baseline}</b>")

  res.append("<h2>Aggregate Costs</h2>")
  res_df = pd.DataFrame(costs)
  fig, axs = plt.subplots(ncols=3, figsize=(18, 6), sharey=True)
  bins = np.arange(0, 1000, 10)
  for ax, cost in zip(axs, ['lataccel_cost', 'jerk_cost', 'total_cost']):
    for controller in ['test', 'baseline']:
      ax.hist(res_df[res_df['controller'] == controller][cost], bins=bins, label=controller, alpha=0.5)
    ax.set_xlabel('Cost')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Cost Distribution: {cost}')
    ax.legend()

  res.append(f'<img src="data:image/png;base64,{img2base64(fig)}" alt="Plot">')
  res.append(res_df.groupby('controller').agg({'lataccel_cost': 'mean', 'jerk_cost': 'mean', 'total_cost': 'mean'}).round(3).reset_index().to_html(index=False))

  res.append("<h2>Sample Rollouts</h2>")
  fig, axs = plt.subplots(ncols=1, nrows=SAMPLE_ROLLOUTS, figsize=(15, 3 * SAMPLE_ROLLOUTS), sharex=True)
  for ax, rollout in zip(axs, sample_rollouts):
    ax.plot(rollout['desired_lataccel'], label='Desired Lateral Acceleration')
    ax.plot(rollout['test_controller_lataccel'], label='Test Controller Lateral Acceleration')
    ax.plot(rollout['baseline_controller_lataccel'], label='Baseline Controller Lateral Acceleration')
    ax.set_xlabel('Step')
    ax.set_ylabel('Lateral Acceleration')
    ax.set_title(f"Segment: {rollout['seg']}")
    ax.axline((CONTROL_START_IDX, 0), (CONTROL_START_IDX, 1), color='black', linestyle='--', alpha=0.5, label='Control Start')
    ax.legend()
  fig.tight_layout()
  res.append(f'<img src="data:image/png;base64,{img2base64(fig)}" alt="Plot">')

  with open("report.html", "w") as fob:
    fob.write("\n".join(res))
    print("Report saved to: './report.html'")

def Cost(x, args):
  tinyphysicsmodel = TinyPhysicsModel(args.model_path, debug=False)
  test_controller = CONTROLLERS[args.test_controller]()
  baseline_controller = CONTROLLERS[args.baseline_controller]()

  data_path = Path(args.data_path)
  assert data_path.is_dir(), "data_path should be a directory"

  costs = []
  sample_rollouts = []
  files = sorted(data_path.iterdir())[:args.num_segs]
  kp = x[0]
  ki = x[1]
  kd = x[2]
  t_c = x[3]
  print(" Kp: " + "{:.3f}".format(kp) + " Ki: " + "{:.3f}".format(ki) + " Kd: " + "{:.3f}".format(kd) + " T_C: " + "{:.3f}".format(t_c))
  print("")
  for d, data_file in enumerate(tqdm(files, total=len(files))):
    test_sim = TinyPhysicsSimulator(tinyphysicsmodel, str(data_file), controller=test_controller, debug=False)
    test_sim.tune(kp, ki, kd, t_c)
    test_cost = test_sim.rollout()
    baseline_sim = TinyPhysicsSimulator(tinyphysicsmodel, str(data_file), controller=baseline_controller, debug=False)
    baseline_cost = baseline_sim.rollout()

    if d < SAMPLE_ROLLOUTS:
      sample_rollouts.append({
        'seg': data_file.stem,
        'test_controller': args.test_controller,
        'baseline_controller': args.baseline_controller,
        'desired_lataccel': test_sim.target_lataccel_history,
        'test_controller_lataccel': test_sim.current_lataccel_history,
        'baseline_controller_lataccel': baseline_sim.current_lataccel_history,
      })
    costs.append({'seg': data_file.stem, 'controller': 'test', **test_cost})
    costs.append({'seg': data_file.stem, 'controller': 'baseline', **baseline_cost})

  # print(costs)
  res = []  
  res_df = pd.DataFrame(costs)
  agg_values = res_df.groupby('controller').agg({'lataccel_cost': 'mean', 'jerk_cost': 'mean', 'total_cost': 'mean'}).round(3).reset_index()
  lat_agg = agg_values.loc[1, 'lataccel_cost']
  jerk_agg = agg_values.loc[1, 'jerk_cost']
  challenge_cost = 5 * lat_agg + jerk_agg

  print("")
  print("Lat: {:.3f}".format(lat_agg) + " jerk: {:.3f}".format(jerk_agg) + " challenge: {:.3f}".format(challenge_cost))
  print("")
  return challenge_cost


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", type=str, required=True)
  parser.add_argument("--data_path", type=str, required=True)
  parser.add_argument("--num_segs", type=int, default=100)
  parser.add_argument("--test_controller", default='pid2', choices=CONTROLLERS.keys())
  parser.add_argument("--baseline_controller", default='simple', choices=CONTROLLERS.keys())
  args = parser.parse_args()

  result = []
  
  bounds = ((0, 2), (0, 1), (0, 1), (0, 1))
  r = minimize(Cost, [0.2, 0.01, 0.05, 0.01],
                args=(args),
                bounds=bounds)
  result.append(r)

        # Print optimization results

  print(" Kp: " + "{:.3f}".format(result[0].x[0]) + " Ki: " + "{:.3f}".format(result[0].x[1]) + " Kd: " + "{:.3f}".format(result[0].x[2]) + " T_C: " + "{:.3f}".format(result[0].x[3]))
  print("")
  print("Success: " + str(r.success))

# https://www.linkedin.com/pulse/how-auto-tune-pid-controller-using-scipy-python-library-sarkar/
