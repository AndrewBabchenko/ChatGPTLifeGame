import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Animal:
    _next_id = 1

    def __init__(self, x, y, name, color, parent_ids=None, predator=False):
        self.id = Animal._next_id
        Animal._next_id += 1
        self.x = x
        self.y = y
        self.name = name
        self.color = color
        self.parent_ids = parent_ids or set()
        self.predator = predator
        self.steps_since_last_meal = 0
        self.mating_cooldown = 0
        self.target = None 
        self.survival_time = 0
        self.num_children = 0

    def move(self, model, animals):
        if self.predator:
            moves = 3 if self.steps_since_last_meal >= steps_since_last_meal else 2
        else:
            moves = 1
        
        visible_animals = self.communicate(animals)
        animal_input = torch.tensor([self.x / 100, self.y / 100, int(self.name == 'A'), int(self.name == 'B')], dtype=torch.float32).unsqueeze(0)
        visible_animals_input = torch.tensor(visible_animals, dtype=torch.float32).unsqueeze(0)
        
        action_prob = model(animal_input, visible_animals_input)

        for _ in range(moves):
            new_x, new_y = self.x, self.y

            if self.predator and self.target:
                dx = self.target.x - self.x
                dy = self.target.y - self.y
                if abs(dx) > abs(dy):
                    new_x += 1 if dx > 0 else -1
                else:
                    new_y += 1 if dy > 0 else -1
            else:
                probabilities = action_prob
                action = torch.multinomial(probabilities, 1).item()

                if action == 0:
                    new_y = (self.y + 1) % 100
                elif action == 1:
                    new_y = (self.y - 1) % 100
                elif action == 2:
                    new_x = (self.x - 1) % 100
                elif action == 3:
                    new_x = (self.x + 1) % 100
                elif action == 4:
                    new_x = (self.x + 1) % 100
                    new_y = (self.y + 1) % 100
                elif action == 5:
                    new_x = (self.x + 1) % 100
                    new_y = (self.y - 1) % 100
                elif action == 6:
                    new_x = (self.x - 1) % 100
                    new_y = (self.y + 1) % 100
                elif action == 7:
                    new_x = (self.x - 1) % 100
                    new_y = (self.y - 1) % 100

            if not self.position_occupied(animals, new_x, new_y):
                self.x, self.y = new_x, new_y

    def move_away(self):
        move_directions = [("up", 0, 1), ("down", 0, -1), ("left", -1, 0), ("right", 1, 0)]
        random.shuffle(move_directions)
        for direction, dx, dy in move_directions:
            new_x = self.x + dx
            new_y = self.y + dy
            if 0 <= new_x <= 99 and 0 <= new_y <= 99:
                self.x = new_x
                self.y = new_y
                break

    def can_mate(self, other):
        return (
            self.name == other.name
            and abs(self.x - other.x) <= 1
            and abs(self.y - other.y) <= 1
            and self.id not in other.parent_ids
            and other.id not in self.parent_ids
            and self.mating_cooldown == 0  
        )

    def can_mate_predator(self):
        return self.predator and self.steps_since_last_meal == 10

    def communicate(self, animals):
        visible_animals = []

        for animal in animals:
            if animal != self:
                dx = abs(animal.x - self.x)
                dy = abs(animal.y - self.y)

                if (dx <= 50 and dy <= 50) or (dx == 0 and dy == 1) or (dx == 1 and dy == 0):
                    visible_animals.append((animal.x / 100, animal.y / 100, int(animal.name == 'A'), int(animal.name == 'B')))

        return visible_animals

    def eat(self, animals):
        if self.predator:
            for i, prey in enumerate(animals):
                if not prey.predator and abs(self.x - prey.x) <= 1 and abs(self.y - prey.y) <= 1:
                    del animals[i]
                    self.steps_since_last_meal = 0
                    return True
            return False

    def display_color(self):
        if self.predator and self.steps_since_last_meal >= steps_since_last_meal:
            return 'darkred'
        return self.color

    def position_occupied(self, animals, new_x, new_y):
        for animal in animals:
            if animal.x == new_x and animal.y == new_y:
                return True
        return False

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 10)  # For the animal's position and type
        self.rnn = nn.GRU(4, 10, batch_first=True)  # For visible animals (x, y, is_prey, is_predator)
        self.fc2 = nn.Linear(20, 8)

    def forward(self, animal_input, visible_animals_input):
        animal_output = F.relu(self.fc1(animal_input))
        _, rnn_output = self.rnn(visible_animals_input)
        rnn_output = rnn_output.squeeze(0)
        combined_output = torch.cat((animal_output, rnn_output), dim=-1)
        return F.softmax(self.fc2(combined_output), dim=-1)


def plot_animals(animals, step):
    plt.clf()
    ax = plt.gca()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title(f"Step: {step}, Preys: {sum(1 for a in animals if a.name == 'A')}, Predators: {sum(1 for a in animals if a.name == 'B')}")
    for animal in animals:
        circle = plt.Circle((animal.x, animal.y), 1, color=animal.display_color())
        ax.add_artist(circle)

    plt.pause(0.0001)

def reward_function(animals, survival_reward_factor=3, children_reward_factor=1):
    prey_reward = 0
    predator_reward = 0

    for animal in animals:
        if animal.predator:
            predator_reward += animal.survival_time * survival_reward_factor * 100
            predator_reward += animal.num_children * children_reward_factor * 100
        else:
            prey_reward += animal.survival_time * survival_reward_factor
            prey_reward += animal.num_children * children_reward_factor

    return prey_reward, predator_reward

                  
def simulate(animals, steps, model_A, model_B, optimizer_A, optimizer_B, mating_probability_A=0.5, mating_probability_B=1, max_animals=300, plot=False, epochs=1):

        total_reward_A = 0
        total_reward_B = 0

        prey_counts = []
        predator_counts = []

        for step in range(steps):
            animals_to_remove = []
            for animal in animals:
                animal.communicate(animals)
                if animal.name == "A":
                    animal.move(model_A, animals)
                else:
                    animal.move(model_B, animals)

                has_eaten = animal.eat(animals)

                # Increase steps_since_last_meal for predators
                if animal.predator and not has_eaten:
                    animal.steps_since_last_meal += 1

                    # Remove the predator if it has not eaten for 100 steps
                    if animal.steps_since_last_meal >= 100:
                        animals_to_remove.append(animal)

            # Remove predators who have not eaten for 30 steps
            for animal in animals_to_remove:
                animals.remove(animal)

            new_animals = []
            mating_pairs = set()
            for i, animal1 in enumerate(animals):
                for j, animal2 in enumerate(animals[i+1:]):
                    if (i, i+1+j) not in mating_pairs and animal1.can_mate(animal2):
                        mating_probability = mating_probability_A if animal1.name == "A" else mating_probability_B
                        if random.random() < mating_probability:
                            mating_pairs.add((i, i+1+j))
                            child_x = (animal1.x + animal2.x) // 2
                            child_y = (animal1.y + animal2.y) // 2
                            child_parent_ids = {animal1.id, animal2.id}.union(animal1.parent_ids, animal2.parent_ids)
                            new_animals.append(Animal(child_x, child_y, animal1.name, animal1.color, child_parent_ids))
                            animal1.move_away()
                            animal2.move_away()
                            animal1.mating_cooldown = 10 
                            animal2.mating_cooldown = 10

                            # Increment the number of children for the parents
                            animal1.num_children += 1
                            animal2.num_children += 1

            if len(animals) + len(new_animals) <= max_animals:
                animals.extend(new_animals)
            else:
                animals.extend(new_animals[:max_animals - len(animals)])

            # Decrease mating cooldown for each animal
            for animal in animals:
                if animal.mating_cooldown > 0:
                    animal.mating_cooldown -= 1

            # Increment survival time for each animal
            for animal in animals:
                animal.survival_time += 1

            prey_count = sum(1 for a in animals if a.name == 'A')
            predator_count = sum(1 for a in animals if a.name == 'B')
            prey_counts.append(prey_count)
            predator_counts.append(predator_count)
            
            reward_A, reward_B = reward_function(animals)
            total_reward_A += reward_A
            total_reward_B += reward_B

            # Update model weights based on the reward signal
            optimizer_A.zero_grad()
            optimizer_B.zero_grad()
            reward_var_A = torch.tensor(reward_A, dtype=torch.float32, requires_grad=True)
            reward_var_B = torch.tensor(reward_B, dtype=torch.float32, requires_grad=True)
            loss_A = -reward_var_A
            loss_B = -reward_var_B
            loss_A.backward()
            loss_B.backward()
            optimizer_A.step()
            optimizer_B.step()

            if plot:
                plot_animals(animals, step)

        return total_reward_A, total_reward_B

def train(animals, episodes, steps, model_A, model_B, optimizer_A, optimizer_B, plot=False):
    episode_rewards = []
    for episode in range(episodes):
        # Create a deep copy of the initial animals for each episode
        episode_animals = [Animal(animal.x, animal.y, animal.name, animal.color, parent_ids=animal.parent_ids, predator=animal.predator) for animal in animals]

        total_reward = simulate(episode_animals, steps, model_A, model_B, optimizer_A, optimizer_B, plot=plot)
        episode_rewards.append(total_reward)


        print(f"Episode {episode}/{episodes}, Total reward: {total_reward}")

    return episode_rewards


# Create instances of the model architectures
loaded_model_A = SimpleNN()
loaded_model_B = SimpleNN()

try:
    # Attempt to load the saved state dictionaries
    loaded_model_A.load_state_dict(torch.load("model_A.pth"))
    loaded_model_B.load_state_dict(torch.load("model_B.pth"))
    print("Models loaded successfully.")
except FileNotFoundError:
    print("Saved models not found. Using empty models.")

loaded_model_A.train()
loaded_model_B.train()

# Initialize the simulation
random.seed()
#torch.manual_seed()

# Configurable parameters
animal_count = 20
field_min = 20
field_max = 80
steps_since_last_meal = 70

# Create 'A' animals as prey
animals = [Animal(random.randint(field_min, field_max), random.randint(field_min, field_max), 'A', 'green') for _ in range(animal_count*2)]

# Create 'B' animals as predators
animals += [Animal(random.randint(field_min, field_max), random.randint(field_min, field_max), 'B', 'red', predator=True) for _ in range(animal_count)]

optimizer_A = optim.Adam(loaded_model_A.parameters(), lr=0.01)
optimizer_B = optim.Adam(loaded_model_B.parameters(), lr=0.01)

# Train the models
episodes = 3
steps = 200
train(animals, episodes, steps, loaded_model_A, loaded_model_B, optimizer_A, optimizer_B, plot=True)

# Set the models to evaluation mode if you're not planning to train them further
loaded_model_A.eval()
loaded_model_B.eval()

torch.save(loaded_model_A.state_dict(), "model_A.pth")
torch.save(loaded_model_B.state_dict(), "model_B.pth")