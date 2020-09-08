function [label, SBR, num_patient] = extract(database_csv, patient_status_csv)

label = true;
label_status = true;
num_patient = zeros;
new_num_patient = zeros;
num_patient_status = zeros;
SBR = zeros(1,4);
new_SBR = zeros(1,4);

for i=1:length(database_csv)
    
    gui = find(database_csv{i} == '"');
    num_patient(i) = str2num(database_csv{i}(1:gui(1)-2));
    SBR(i,:) = [str2num(database_csv{i}(gui(3)+1:gui(4)-1)) str2num(database_csv{i}(gui(5)+1:gui(6)-1)) str2num(database_csv{i}(gui(7)+1:gui(8)-1)) str2num(database_csv{i}(gui(9)+1:gui(10)-1))];
    
end

new_num_patient = unique(num_patient); 

for i=1:length(new_num_patient)
    temp = SBR(new_num_patient(i)==num_patient,:); 
    new_SBR(i,:)=temp(1,:);
end



for i=1:length(patient_status_csv)
    
    if patient_status_csv{i}(gui(1)+1) == 'H'
        label_status(i) = false;
    else
        label_status(i) = true;
    end
    
    num_patient_status(i) = str2num(patient_status_csv{i}(1:gui(1)-2));

end

for i=1:length(new_num_patient)
    
    if sum(new_num_patient(i) == num_patient_status)~=0
        
        label(i) = label_status(new_num_patient(i) == num_patient_status);
        
    end
    
end

SBR = new_SBR;
num_patient = new_num_patient;

end