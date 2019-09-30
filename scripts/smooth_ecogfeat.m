%% make the MNI grid
[x,y,z] = meshgrid(0:255,0:255,0:255);
grid2 = zeros(numel(x),3);
for i=1:numel(x)
    grid2(i,:) = [x(i),y(i),z(i)];
end

%% load the tau and coordinates
load /Users/rdgao/Documents/code/research/field-echos/results/MNI_rest/mni_tau_coor.mat
vin = elec_ijk(:,1);
grid1 = elec_ijk(:,2:4);

%% do spatial smoothing
% this takes a long ass time since the coordinates are really dense

% d happens to be in the same units of MNI coordinates since the affine
% transform has unity scaling
d = 5; 
vout = spatfiltergauss(vin, grid1, d, grid2);

%% save out
