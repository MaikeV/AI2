import math
from pomegranate import *

pEarthquake = DiscreteDistribution({'E': 0.002, 'NE': 0.998})
pBurglary = DiscreteDistribution({'B': 0.001, 'NB': 0.999})

cpAlarm = ConditionalProbabilityTable([
    # [Burglary, Earthquake, Alarm, Probability]
    ['B', 'E', 'A', 0.95],
    ['B', 'E', 'NA', 0.05],
    ['B', 'NE', 'A', 0.94],
    ['B', 'NE', 'NA', 0.06],

    ['NB', 'E', 'A', 0.29],
    ['NB', 'E', 'NA', 0.71],
    ['NB', 'NE', 'A', 0.001],
    ['NB', 'NE', 'NA', 0.999],
], [pBurglary, pEarthquake])

cpJohnCalling = ConditionalProbabilityTable([
    # [Alarm, John, Probability]
    ['A', 'J', 0.9],
    ['A', 'NJ', 0.1],

    ['NA', 'J', 0.05],
    ['NA', 'NJ', 0.95],
], [cpAlarm])

cpMaryCalling = ConditionalProbabilityTable([
    # [Alarm, Mary, Probability]
    ['A', 'M', 0.7],
    ['A', 'NM', 0.3],

    ['NA', 'M', 0.01],
    ['NA', 'NM', 0.99],
], [cpAlarm])

earthquakeState = State(pEarthquake, name="earthquakeState")
burglaryState = State(pBurglary, name="burglaryState")
alarmState = State(cpAlarm, name="alarmState")
johnCallingState = State(cpJohnCalling, name="johnCallingState")
maryCallingState = State(cpMaryCalling, name="maryCallingState")

network = BayesianNetwork("Solving the Monty Hall Problem with bayesian network")

network.add_states(earthquakeState, burglaryState, alarmState, johnCallingState, maryCallingState)

network.add_edge(earthquakeState, alarmState)
network.add_edge(burglaryState, alarmState)
network.add_edge(alarmState, johnCallingState)
network.add_edge(alarmState, maryCallingState)

network.bake()

print("############### John or Mary Calling given Alarm ###############")
beliefs = network.predict_proba({'alarmState': 'A'})
beliefs = map(str, beliefs)
print("\n".join("{}\t{}".format(state.name, belief) for state, belief in zip(network.states, beliefs)))

print("############### John Calling without Alarm ###############")
beliefs = network.predict_proba({'alarmState': 'NA'})
beliefs = map(str, beliefs)
print("\n".join("{}\t{}".format(state.name, belief) for state, belief in zip(network.states, beliefs)))

print("############### Mary Calling after an Earthquake ###############")
beliefs = network.predict_proba({'earthquakeState': 'E'})
beliefs = map(str, beliefs)
print("\n".join("{}\t{}".format(state.name, belief) for state, belief in zip(network.states, beliefs)))

print("############### Alarm goes off without Burglary or Earthquake and neither John nor Mary Calling ###############")
beliefs = network.predict_proba(
    {'johnCallingState': 'NJ', 'maryCallingState': 'NM', 'earthquakeState': 'NE', 'burglaryState': 'NB'})
beliefs = map(str, beliefs)
print("\n".join("{}\t{}".format(state.name, belief) for state, belief in zip(network.states, beliefs)))
