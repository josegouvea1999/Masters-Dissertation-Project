import { Text, ScrollView, View, Image, Pressable, Modal, ActivityIndicator, Animated, Easing, Linking, TouchableOpacity } from "react-native";
import React ,{useEffect, useState, useRef} from "react";
import { useNavigation } from "@react-navigation/native";
import {styles} from "./Styles"
import { LinearGradient } from "expo-linear-gradient";
import axios from "axios";

const SERVER_URL = "http://192.168.1.70:5000"

const LoadingScreen = () => (
    <View style={styles.loadingContainer}>
      <Image source={require("../assets/logo.png")} style={styles.loadingLogo} />
      <Text style={styles.loadingText}>Loading your Spotify data...</Text>
      
    </View>
);

const PulsatingImage = () => {

    const scaleValue = useRef(new Animated.Value(1)).current;
  
    useEffect(() => {
      const pulse = () => {
        Animated.sequence([
          Animated.timing(scaleValue, {
            toValue: 1.2,
            duration: 1000,
            easing: Easing.linear,
            useNativeDriver: true,
          }),
          Animated.timing(scaleValue, {
            toValue: 1,
            duration: 1000,
            easing: Easing.linear,
            useNativeDriver: true,
          }),
        ]).start(() => pulse());
      };
  
      pulse();
    }, [scaleValue]);
  
    return (
      <View style={styles.overlayContainer}>
        <Animated.Image
          source={require('../assets/logo_filled.png')}
          style={[styles.pulsatingImage, { transform: [{ scale: scaleValue }] }]}
        />
      </View>
    );
  };

const HomeScreen = () => {

    const navigation = useNavigation();

    const [loading, setLoading] = useState(true);
    const [message, setMessage] = useState(null);
    const [userProfile, setUserProfile] = useState(null);
    const [modalVisible, setModalVisible] = useState(false);
    const [selectedPlaylist, setSelectedPlaylist] = useState(null);
    const [radio_url, setRadioUrl] = useState(null);
    const [radioedPlaylists, setRadioedPlaylists] = useState([{}]);
    const [radioedPlaylistCovers, setRadioedPlaylistsCovers] = useState([]);
    const [playlists, setPlaylists] = useState([{}]);
    const [playlistCovers, setPlaylistCovers] = useState([]);


    const [genInProgress, setGenInProgress] = useState(false);
    const [genCompleted, setGenCompleted] = useState(false);
    const [modalHeight, setModalHeight] = useState(300);


    useEffect(() => {
        const fetchData = async () => {
          await Promise.all([getProfile(), greetingMessage(), getPlaylists()]);
          setLoading(false);
        };
    
        fetchData();
      }, []);

    const greetingMessage = () => {
        const currentTime = new Date().getHours();
        if (currentTime < 12) {
          setMessage("Good Morning");
        } else if (currentTime < 16) {
          setMessage("Good Afternoon");
        } else {
          setMessage("Good Evening");
        }
    };

    const getProfile = async () => {
        try {
          const response = await axios.get( SERVER_URL + "/profile", {withCredentials : true} );
          const data = await response.data;
          console.log("User logged in:");
          console.log(data);
    
          setUserProfile(data);
    
        } catch (error) {
          console.log(error);
        }
    }

    const getPlaylists = async () => {
        try {
          var response = await axios.get(SERVER_URL + "/tastes_playlists", {withCredentials : true} );
          var radioed_data = await response.data;
          
          console.log(radioed_data);

          if (radioed_data != []) {
            const radioed_cover_urls = radioed_data.map(item => (item.images && item.images[0] && item.images[0].url) ? item.images[0].url : null);
            setRadioedPlaylists(radioed_data)
            setRadioedPlaylistsCovers(radioed_cover_urls);   
          }

          
          response = await axios.get(SERVER_URL + "/playlists", {withCredentials : true} );
          var playlist_data = await response.data.items;

          radioed_data = radioed_data.map((item) => item.id);
          playlist_data = playlist_data.filter(item => !radioed_data.includes(item.id));

          const cover_urls = playlist_data.map(item => (item.images && item.images[0] && item.images[0].url) ? item.images[0].url : null);

          setPlaylists(playlist_data);

          setPlaylistCovers(cover_urls);   
          
        } catch (error) {
          console.log(error);
        } 
      }

    const setModal = async (playlist_id, playlist_name, playlist_cover, playlist_url) => {

      setSelectedPlaylist({playlist_id, playlist_name, playlist_cover, playlist_url});
      setModalVisible(true);
    }

    const genRecommendations = async () => {
      try {
        setModalHeight(600);
        setGenInProgress(true);
  
        console.log("Refreshing " + selectedPlaylist.playlist_name + "'s RADIO playlist");
  
        const response = await axios.get( SERVER_URL + "/refresh/" + selectedPlaylist.playlist_id, {withCredentials : true} );
        
        const data = await response.data;
  
        setRadioUrl(data.external_urls.spotify);
        setGenInProgress(false);
        setGenCompleted(true);

  
        console.log(selectedPlaylist.playlist_name + "'s RADIO has been refreshed successfully");
        
      } catch (error) {
        console.log(error);
      }
    }

    const resetModal = async () => {

      setModalVisible(false);
      setGenCompleted(false);
      setModalHeight(300);
    }

    return (  

        <LinearGradient colors={["#040306", "#131624"]} style={{ flex: 1 }}>
  
          {loading ? ( <LoadingScreen /> ) : ( 
              <View>
                <View style={styles.userBar}>
                  <Image
                    style={styles.userIcon}
                    source={{ uri: userProfile?.images[0]?.url }}/>
                  <View style={{maxWidth: '60%'}}>
                    <Text style={styles.userMessage}>
                      {message +  " " + userProfile?.display_name.split(" ", 2)[0] + "!"}
                    </Text>
                    <Text style={styles.selectPlaylistMessage}>
                      {"Please select the playlist whose RADIO you wish to refresh..."}
                    </Text>
                  </View>
                  <Image
                    source={require('../assets/logo.png')}
                    style={{ width: 70, height: 60, marginLeft: 30}} />
                </View>
            
                <ScrollView style={{ marginTop: 10, marginBottom: 150 }}>
            
                    <Modal
                      animationType="slide"
                      transparent={true}
                      visible={modalVisible}
                      onRequestClose={() => setModalVisible(!modalVisible)}>
                      <View style={styles.centeredView}>
                        <LinearGradient colors={["#1DB954", "#131624"]} style={[styles.modalView, {height: modalHeight}]}>
                          {genInProgress ? (
                                selectedPlaylist && (
                                    <View style={styles.normalContent}>
                                        <View style={[{flexDirection: 'row', alignItems: 'center', justifyContent: 'center', marginTop: 100}]}>
                                            <Image source={{ uri: selectedPlaylist.playlist_cover }} style={styles.playlistImageModalExpanding} />
                                            <PulsatingImage style={{position: 'absolute', left:'50%'}}/>
                                        </View>
                                        <Text style={[styles.textStyleExpanding, styles.modalTextExpanding]}>
                                            {`Refreshing "${selectedPlaylist.playlist_name}"'s RADIO playlist...`}
                                        </Text>
                                        <ActivityIndicator size="large" color="white"/>
                                    </View>
                                )
                            ) : (
                                genCompleted ? selectedPlaylist && (
                                    <View style={styles.normalContent}>
                                        <Text style={styles.textStyleSuccess}>
                                            {"NEW RECOMMENDATIONS GENERATED SUCCESSFULLY!"}
                                        </Text>
                                        <Text style={styles.textStyleSuccess2}>
                                            Listen to your new recommendations in the designated [RADIO] playlist in your Spotify library:
                                        </Text>
                                        <Image source={require('../assets/logo.jpg')} style={styles.playlistImageModalSuccess} />
                                        <Text style={styles.textStyleSuccess3}>
                                            {selectedPlaylist.playlist_name} [RADIO]
                                        </Text>
                                        <View style={styles.buttonContainerSuccess}>
                                            <Pressable
                                                style={[
                                                    styles.buttonSuccess,
                                                    styles.buttonClose,
                                                    styles.confirmButtonSuccess,
                                                    ]}
                                                onPress={() => Linking.openURL(radio_url)}>
                                                <Text style={[styles.textStyle, styles.confirmButtonText]}>Go to playlist [RADIO]</Text>
                                            </Pressable>
                                            <Pressable
                                                style={[
                                                    styles.buttonSuccess,
                                                    styles.buttonClose,
                                                    styles.cancelButton,
                                                ]}
                                                onPress={() => resetModal()}>
                                                <Text style={[styles.textStyle, {fontSize: 14}]}>Get more recommendations...</Text>
                                            </Pressable>
                                        </View>
                                    </View>
                                ) : (
                                    selectedPlaylist && (
                                        <View style={styles.normalContent}>
                                            <Text style={[styles.textStyle, styles.modalText]}>
                                                {`Do you wish to refresh this playlist's RADIO?`}
                                            </Text>
                                            <View style={styles.playlistInfoContainer}>
                                                <Image
                                                    source={{ uri: selectedPlaylist.playlist_cover }}
                                                    style={styles.playlistImageModal}/>
                                                <Text style={styles.playlistNameModal}>
                                                    {selectedPlaylist.playlist_name}
                                                </Text>
                                            </View>
                                            <View style={styles.buttonContainer}>
                                                <Pressable
                                                    style={[
                                                        styles.button,
                                                        styles.buttonClose,
                                                        styles.cancelButton,
                                                    ]}
                                                    onPress={() => setModalVisible(!modalVisible)}>
                                                    <Text style={[styles.textStyle, styles.cancelButtonText]}>Cancel</Text>
                                                </Pressable>
                                                <Pressable
                                                    style={[
                                                        styles.button,
                                                        styles.buttonClose,
                                                        styles.confirmButton,
                                                    ]}
                                                    onPress={() => genRecommendations()}>
                                                    <Text style={[styles.textStyle, styles.confirmButtonText]}>Confirm</Text>
                                                </Pressable>
                                            </View>
                                        </View>
                                    )
                                )
                            )}
                        </LinearGradient>
                      </View>
                    </Modal>
                                                
                    {radioedPlaylists == [{}] ? (
                        <View>
                            <Text style={styles.textStyleSuccess2}>
                                You have no radio playlists...
                            </Text>
                        </View>
                    ) : (
                      radioedPlaylists.map((playlist, i) => (
                            <Pressable
                                key={i}
                                onPress={() => setModal(playlist.id, playlist.name, radioedPlaylistCovers[i], playlist.external_urls.spotify)}
                                style={styles.playlistPressable}>
                                <LinearGradient colors={["#33006F", "#FFFFFF"]}>
                                    <Pressable style={styles.coverPressable}>
                                        <Image
                                            source={{ uri: radioedPlaylistCovers[i]  }}
                                            style={{ width: 75, height: 75 }}/>
                                    </Pressable>
                                </LinearGradient>
                                <Text style={{ color: "white", fontSize: 15, fontWeight: "bold", maxWidth: '75%' }}>
                                    {playlist.name}
                                </Text>
                            </Pressable>
                        ))
                    )}

                  <View style={styles.container}>
                    <TouchableOpacity
                      style={styles.addButton}
                      onPress={() => navigation.navigate('Add', { playlists, playlistCovers })}
                    >
                      <Text style={styles.addButtonText}>+</Text>
                    </TouchableOpacity>
                  </View>
                
                </ScrollView>
              </View>)}
          </LinearGradient>
    );
};

export default HomeScreen;