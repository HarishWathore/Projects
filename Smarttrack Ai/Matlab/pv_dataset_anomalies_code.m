% Given values
Ta_values = 0:50;
Gir_values = 0:1000;
alpha = -0.3;
beta = -0.05;
T0 = 25;
G0 = 1000;
V0 = 22;
I0 = 1.955;

% Function to calculate Tm
calculate_Tm = @(Gir, Ta) 30 - 0.0175 * (Gir - 300) + 1.14 * (Ta - 25);

% Function to calculate Vth
calculate_Vth = @(Tm, reducedV0) reducedV0 * (1 + beta * (Tm - T0));

% Function to calculate In0
calculate_In0 = @(Tm, Gir) I0 * (1 + alpha * (Tm - T0)) * (Gir / G0);

% Function to calculate Pac
calculate_Pac = @(Vth, In0) Vth * In0;

% Generate dataset
dataset = cell(0, 1);

for Ta = Ta_values
    for Gir = Gir_values
        % Calculate Tm separately
        Tm = calculate_Tm(Gir, Ta);
        
        % Calculate Vth separately with reduced V0 for anomalies
        reducedV0 = V0;
        if rand() < 0.05
            reducedV0 = 2.2;  % Reduce V0 to 2.2 for anomalies
        end
        Vth = calculate_Vth(Tm, reducedV0);
        
        % Calculate In0 separately
        In0 = calculate_In0(Tm, Gir);
        
        % Calculate Pac
        Pac = calculate_Pac(Vth, In0);
        
        % Append values to the dataset
        entry.Ta = Ta;
        entry.Gir = Gir;
        entry.Tm = Tm;
        entry.Vth = Vth;
        entry.In0 = In0;
        entry.Pac = Pac;
        dataset = [dataset; entry];
    end
end

% Convert the cell array to a table
dataTable = struct2table([dataset{:}]);

% Save the table to a CSV file
writetable(dataTable, 'PV_dataset_anomalies.csv');
disp('Dataset with anomalies (V0 reduced to 2.2) and Pac calculated saved to PV_dataset_anomalies.csv');
