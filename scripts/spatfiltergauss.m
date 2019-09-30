% function kindly provided by Thomas Pfeffer
function vout=spatfiltergauss(vin,grid,d,grid2);
% makes a spatialfilter with a Gaussian function
% usage: vout=spatfiltergauss(vin,grid,d)
% or:   vout=spatfiltergauss(vin,grid,d,grid2)
%
% if only one grid is provide the smoothed field is calcuated on the
% the same grid as for original field
%
% input: vin   NxK matrix, for N voxels and K fields, each column is a field (e.g. power)
%              defined on grid provided as the second argument
%        grid  Nx3 matrix, grid locations of original data,
%        d     width of Gaussian, at distance d the Gaussian drops to 50%
%                of its maximum
%        grid2 (optional) Mx3 matrix of grid locations where the smoothed field is calculated
%               if not provided, it is set to grid.
%
% output: vout NxK (or MxK, if grid2 is provided) matrix of smoothed
%               fields.
%

if nargin==3;

[ng ns]=size(vin);
alpha=d/sqrt(-log(.5));
vout=vin;
for i=1:ng
    r0=grid(i,:);
    rd=grid-repmat(r0,ng,1);
    dis=sqrt(sum(rd.^2,2));
    w=exp(-dis.^2/alpha^2);
    for j=1:ns
    vout(i,j)=sum(w.*vin(:,j))/sum(w);
    end
end

elseif nargin==4

    [ng ns]=size(vin);
    alpha=d/sqrt(-log(.5));

    [ng2 ndum]=size(grid2);
    vout=zeros(ng2,ns);


    for i=1:ng2
        r0=grid2(i,:);
        rd=grid-repmat(r0,ng,1);
        dis=sqrt(sum(rd.^2,2));
        w=exp(-dis.^2/alpha^2);
        for j=1:ns
            vout(i,j)=sum(w.*vin(:,j))/sum(w);
        end
    end

end



return
