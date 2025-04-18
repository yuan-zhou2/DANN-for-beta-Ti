import numpy as np
import pandas as pd

# Names of elements
Name = ['Ti', 'Nb', 'Zr', 'Sn', 'Mo', 'Ta']

# Average mixing enthalpy
data_path = r"H.xlsx"
enthalpy = pd.read_excel(data_path, index_col=0)

R = 8.314  # J/(mol·K)

# Atomic radii (pm)
atomic_radii = {'Ti': 147, 'Nb': 146, 'Zr': 160, 'Sn': 145, 'Mo': 139, 'Ta': 143}

# Average valence electrons
average_ea = {
    'Ti': 4,
    'Nb': 5,
    'Zr': 4,
    'Sn': 4,
    'Mo': 6,
    'Ta': 5
}

# Molybdenum equivalent coefficients
eq_weight = {
    'Ti': 0,
    'Nb': 0.33,
    'Zr': 0.31,
    'Sn': 0.3,
    'Mo': 1,
    'Ta': 0.25
}

# Electronegativity values
electronegativity = {
    'Ti': 1.54,
    'Nb': 1.6,
    'Zr': 1.33,
    'Sn': 1.96,
    'Mo': 2.16,
    'Ta': 1.5
}

# Relative atomic masses
relative_atomic_mass = {
    'Ti': 47.867,
    'Nb': 92.906,
    'Zr': 91.224,
    'Sn': 118.710,
    'Mo': 95.95,
    'Ta': 180.947,
}


def mass_percent_to_atomic_percent(mass_percents, atomic_masses):
    total = 0
    percents = {}
    for name in Name:
        total += mass_percents[name] / atomic_masses[name]
    for name in Name:
        percents[name] = (mass_percents[name] / atomic_masses[name]) / total * 100
    return pd.DataFrame(percents)


class Parameter_Calculate():
    def __init__(self):
        self.Name = Name
        self.enthalpy = enthalpy
        self.average_ea = average_ea
        self.eq_weight = eq_weight
        self.electronegativity = electronegativity
        self.relative_atomic_mass = relative_atomic_mass
        self.atomic_radii = atomic_radii
        self.R = R

    def calculate_count(self, df):
        percents = mass_percent_to_atomic_percent(df, self.relative_atomic_mass)
        for name in self.Name:
            df[f'have_{name}'] = (percents[name] > 0.5).astype(int)
        return df

    def calculate_count_num(self, df):
        percents = mass_percent_to_atomic_percent(df, self.relative_atomic_mass)
        df['count'] = 0
        for name in self.Name:
            df['count'] += (percents[name] > 0.5).astype(int)
        return df

    def binary_encode(self, df):
        # Get the name of the last column
        last_column = df.columns[-1]
        # Get data from the last column
        numbers = df[last_column]

        # Determine the required number of binary bits
        num_bits = 3

        binary_data = []
        for number in numbers:
            # Subtract 1 and convert to binary string, padded to num_bits
            binary_str = format(number - 1, f'0{num_bits}b')
            binary_list = [int(bit) for bit in binary_str]
            binary_data.append(binary_list)

        # Convert binary list to DataFrame
        encoded_df = pd.DataFrame(binary_data)
        # Generate column names for encoded bits
        encoded_columns = [f'{last_column}_bit_{i + 1}' for i in range(num_bits)]
        encoded_df.columns = encoded_columns

        # Concatenate original DataFrame (without last column) with encoded bits
        df = pd.concat([df.drop(columns=[last_column]), encoded_df], axis=1)
        return df

    # Average mixing entropy
    def calculate_Average_mixing_entropy(self, df):
        percents = mass_percent_to_atomic_percent(df, self.relative_atomic_mass)
        df['Average_mixing_entropy'] = percents.apply(self._row_Average_mixing_entropy, axis=1)
        return df

    def _row_Average_mixing_entropy(self, row):
        mix_entropy = 0
        for name in self.Name:
            if row[name] > 0:
                p = row[name] / 100
                mix_entropy += p * np.log(p)
        return -self.R * mix_entropy

    # Calculate atomic radius difference
    def calculate_Atomic_radius_difference(self, df):
        percents = mass_percent_to_atomic_percent(df, self.relative_atomic_mass)
        avg_radius = sum(percents[name] * self.atomic_radii[name] for name in self.Name) / 100
        variance = sum(percents[name] * (1 - self.atomic_radii[name] / avg_radius) ** 2 for name in self.Name) / 100
        df['Atomic_radius_difference'] = np.sqrt(variance)
        return df

    # Calculate e/a parameter
    def calculate_ea(self, df):
        percents = mass_percent_to_atomic_percent(df, self.relative_atomic_mass)
        e_a = sum(self.average_ea[name] * percents[name] for name in self.Name)
        df['e/a'] = e_a / 100
        return df

    # Calculate Mo_eq
    def calculate_Moeq(self, df):
        Mo_eq = sum(self.eq_weight[name] * df[name] for name in self.Name)
        df['Mo_eq'] = Mo_eq
        return df

    # Calculate cluster formula
    def calculate_cluster(self, df):
        percents = mass_percent_to_atomic_percent(df, self.relative_atomic_mass)
        M1 = np.array([16, 17, 18])
        M2 = np.array([18, 19, 20])
        k = np.zeros(len(df), dtype=bool)

        # Condition 1: Mo > 0
        mask1 = df['Mo'] > 0

        # For rows satisfying condition 1
        if np.any(mask1):
            for m in M1:
                u = percents['Mo'] * m / 100
                v = percents['Sn'] * m / 100
                w = percents['Ti'] * m / 100
                x = percents['Nb'] * m / 100
                y = percents['Ta'] * m / 100
                z = percents['Zr'] * m / 100
                cond1 = (
                    (u + v) >= 1
                    & (x + y + z) >= (m - 15)
                    & (u >= 0)
                    & (v >= 0)
                    & (w >= 0)
                    & (z >= 0)
                    & (x >= 0)
                    & (y >= 0)
                )
                k = np.logical_or(k, cond1)

        # For rows not satisfying condition 1
        if np.any(~mask1):
            for m in M2:
                v = percents['Sn'] * m / 100
                w = percents['Ti'] * m / 100
                z1 = percents['Zr'] * m / 100
                x = percents['Nb'] * m / 100
                y = percents['Ta'] * m / 100
                # Subcondition 2 for condition 2
                cond2 = (
                    (v + z1) >= 1
                    & (x + y) >= (m - 15)
                    & (v >= 0)
                    & (w >= 0)
                    & (z1 >= 0)
                    & (x >= 0)
                    & (y >= 0)
                )
                k = np.logical_or(k, cond2)

        # Write results back to DataFrame
        df['is_the_best_cluster'] = np.where(k, 0, 1)
        return df

    # Average electronegativity
    def create_Average_electronegativity(self, df):
        percents = mass_percent_to_atomic_percent(df, self.relative_atomic_mass)
        avg_en = sum(self.electronegativity[name] * percents[name] for name in self.Name)
        df['Average_electronegativity'] = avg_en / 100
        return df

    # Relative atomic mass difference
    def create_Relative_atomic_mass(self, df):
        percents = mass_percent_to_atomic_percent(df, self.relative_atomic_mass)
        avg_mass = sum(self.relative_atomic_mass[name] * percents[name] for name in self.Name) / 100
        rel_mass = sum(abs(self.relative_atomic_mass[name] - avg_mass) * percents[name] for name in self.Name) / 100
        df['relative_atomic_mass'] = rel_mass
        return df

    # Enthalpy of mixing
    def create_enthalpy_of_mixing(self, df):
        percents = mass_percent_to_atomic_percent(df, self.relative_atomic_mass)
        mix_enthalpy = 0
        for i in self.Name:
            for j in self.Name:
                mix_enthalpy += 4 * self.enthalpy.loc[i, j] * percents[i] * percents[j]
        df['enthalpy_of_mixing'] = mix_enthalpy / 10000 / 2
        return df

parameter_calculator = Parameter_Calculate()

def calculate_params(df):
    df = parameter_calculator.calculate_ea(df)
    df = parameter_calculator.calculate_Moeq(df)
    df = parameter_calculator.create_Average_electronegativity(df)
    df = parameter_calculator.calculate_Atomic_radius_difference(df)
    df = parameter_calculator.calculate_count(df)
    df = parameter_calculator.calculate_count_num(df)
    df = parameter_calculator.binary_encode(df)
    return df
