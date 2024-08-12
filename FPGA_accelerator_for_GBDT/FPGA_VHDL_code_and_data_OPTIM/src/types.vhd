-------------------------------------------------------------------------------
-- Types and functions
-------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.math_real."ceil";
use IEEE.math_real."log2";

package types is
    -- Signals size
    function log_2(n: natural) return natural;
end package;

package body types is
    -- Signals size
    function log_2(n: natural) return natural is
    begin
        return integer(ceil(log2(real(n))));
    end function;
end types;

