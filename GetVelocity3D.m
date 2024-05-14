function [Velocities] = GetVelocity3D(x)
    Velocities = zeros(1, size(x, 2));
    for i = 2:size(x, 2)
        xyz1 = x(:, i);
        xyz0 = x(:, i-1);
        DeltaX = (xyz1(1) - xyz0(1))*(xyz1(1) - xyz0(1));
        DeltaY = (xyz1(2) - xyz0(2))*(xyz1(2) - xyz0(2));
        DeltaZ = (xyz1(3) - xyz0(3))*(xyz1(3) - xyz0(3));
        DeltaXYZ = sqrt(DeltaX + DeltaY + DeltaZ);
        Velocities(1, i) = DeltaXYZ;
    end
end

