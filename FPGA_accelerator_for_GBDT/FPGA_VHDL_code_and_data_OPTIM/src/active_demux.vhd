-------------------------------------------------------------------------------
-- Generic demultiplexer with activation control
-------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.types.all;

entity active_demux is
    generic(OUTPUT_LENGTH:  natural);
    port(Active: in std_logic;
         Sel:    in std_logic_vector(log_2(OUTPUT_LENGTH) - 1 downto 0);
         Dout:   out std_logic_vector(OUTPUT_LENGTH - 1 downto 0));
end active_demux;

architecture Behavioral of active_demux is
begin
    process (Active, Sel)
    begin
        if Active = '1' then
            Dout <= (others => '0');
            Dout(to_integer(unsigned(Sel))) <= '1';
        else
            Dout <= (others => '0');
        end if;
    end process;
end Behavioral;

