import math
from pomegranate import *

pGuestChoice = DiscreteDistribution({'A': 1. / 3, 'B': 1. / 3, 'C': 1. / 3})
pPrizeDoor = DiscreteDistribution({'A': 1. / 3, 'B': 1. / 3, 'C': 1. / 3})

cpMontysChoice = ConditionalProbabilityTable([
        #[guestChoice, prizeDoor, montysPick, probability]
        ['A', 'A', 'A', 0.0],
        ['A', 'A', 'B', 0.5],
        ['A', 'A', 'C', 0.5],
        ['A', 'B', 'A', 0.0],
        ['A', 'B', 'B', 0.0],
        ['A', 'B', 'C', 1.0],
        ['A', 'C', 'A', 0.0],
        ['A', 'C', 'B', 1.0],
        ['A', 'C', 'C', 0.0],

        ['B', 'A', 'A', 0.0],
        ['B', 'A', 'B', 0.0],
        ['B', 'A', 'C', 1.0],
        ['B', 'B', 'A', 0.5],
        ['B', 'B', 'B', 0.0],
        ['B', 'B', 'C', 0.5],
        ['B', 'C', 'A', 1.0],
        ['B', 'C', 'B', 0.0],
        ['B', 'C', 'C', 0.0],

        ['C', 'A', 'A', 0.0],
        ['C', 'A', 'B', 1.0],
        ['C', 'A', 'C', 0.0],
        ['C', 'B', 'A', 1.0],
        ['C', 'B', 'B', 0.0],
        ['C', 'B', 'C', 0.0],
        ['C', 'C', 'A', 0.5],
        ['C', 'C', 'B', 0.5],
        ['C', 'C', 'C', 0.0],
    ], [pGuestChoice, pPrizeDoor]
)

guestChoiceState = State(pGuestChoice, name="guestChoiceState")
prizeDoorState = State(pPrizeDoor, name="prizeDoorState")
montysChoiceState = State(cpMontysChoice, name="montysChoiceState")

network = BayesianNetwork("Solving the Monty Hall Problem with bayesian network")

network.add_states(guestChoiceState, prizeDoorState, montysChoiceState)

network.add_edge(guestChoiceState, montysChoiceState)
network.add_edge(prizeDoorState, montysChoiceState)

network.bake()

beliefs = network.predict_proba({'guestChoiceState' : 'A'})
beliefs = map(str, beliefs)
print("\n".join("{}\t{}".format(state.name, belief) for state, belief in zip(network.states, beliefs)))

beliefs = network.predict_proba({'guestChoiceState' : 'A', 'montysChoiceState' : 'C'})
beliefs = map(str, beliefs)
print("\n".join("{}\t{}".format(state.name, belief) for state, belief in zip(network.states, beliefs)))

