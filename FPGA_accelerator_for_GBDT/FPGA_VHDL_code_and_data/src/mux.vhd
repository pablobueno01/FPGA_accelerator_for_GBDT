-------------------------------------------------------------------------------
-- Generic multiplexer
-------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.types.all;

entity mux is
    generic(DATA_LENGTH: natural;
            NUM_INPUTS:  natural);
    port(Ctrl: in std_logic_vector(log_2(NUM_INPUTS) - 1 downto 0);
         Din:  in std_logic_vector(NUM_INPUTS * DATA_LENGTH - 1 downto 0);
         Dout: out std_logic_vector(DATA_LENGTH - 1 downto 0));
end mux;

architecture Behavioral of mux is
begin
    Dout <= Din((to_integer(unsigned(Ctrl)) + 1) * DATA_LENGTH - 1
                downto to_integer(unsigned(Ctrl)) * DATA_LENGTH);
end Behavioral;

